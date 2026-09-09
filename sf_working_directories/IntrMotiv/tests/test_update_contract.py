from types import SimpleNamespace

import torch

from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import (
    TargetFiLMDecoder,
    controller_core_view,
)
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu
from sf_working_directories.IntrMotiv.dmlab.custom_learner import (
    BaseDistanceRecorder,
    DistanceLearnerReward,
    categorical_action_total_variation,
    complete_rollout_generation_mask,
    dg_gradient_interaction_stats,
    finite_masked_mean,
    forced_recruitment_preflight_selection,
    legacy_reward_streams,
    normalize_dg_projection_rows,
    record_poststep_calibration_count,
    require_sufficient_generation_batch,
)


def test_categorical_action_total_variation_is_probability_space_and_bounded():
    logits = torch.tensor([[0.0, 0.0], [10.0, -10.0]])
    alternate = torch.tensor([[0.0, 0.0], [-10.0, 10.0]])

    total_variation = categorical_action_total_variation(logits, alternate)

    assert total_variation[0].item() == 0.0
    assert 0.999 < total_variation[1].item() <= 1.0


def test_finite_masked_mean_ignores_no_route_sentinels_and_is_finite_when_empty():
    values = torch.tensor([float("inf"), 2.0, 4.0, float("nan")])
    assert finite_masked_mean(values, torch.tensor([True, True, True, True])).item() == 3.0
    empty = finite_masked_mean(values, torch.tensor([True, False, False, True]))
    assert torch.isfinite(empty) and empty.item() == 0.0


def test_forced_recruitment_preflight_waits_selects_once_and_uses_first_valid():
    valids = torch.tensor([False, True, False, True])
    assert forced_recruitment_preflight_selection(valids, -1, 16, 10, 4, 0) is None
    assert forced_recruitment_preflight_selection(valids, 3, 16, 3, 4, 0) is None
    assert forced_recruitment_preflight_selection(valids, 3, 16, 4, 4, 0) == (1, 3)
    assert forced_recruitment_preflight_selection(valids, 3, 16, 10, 4, 1) is None


def test_controller_backward_stops_ca3_but_keeps_bypass_gradient():
    ca3 = torch.randn(5, 4, requires_grad=True)
    bypass = torch.randn(5, 3, requires_grad=True)
    decoder = torch.nn.Linear(7, 2)

    decoder(controller_core_view(torch.cat((ca3, bypass), dim=-1), 4)).sum().backward()

    assert ca3.grad is None or torch.equal(ca3.grad, torch.zeros_like(ca3))
    assert bypass.grad is not None and bypass.grad.abs().sum() > 0
    assert decoder.weight.grad is not None and decoder.weight.grad.abs().sum() > 0


def test_controller_joint_mode_preserves_outputs_and_allows_ca3_gradient():
    ca3 = torch.randn(5, 4, requires_grad=True)
    bypass = torch.randn(5, 3, requires_grad=True)
    combined = torch.cat((ca3, bypass), dim=-1)
    stopped = controller_core_view(combined, 4, "stop")
    joint = controller_core_view(combined, 4, "joint")

    assert torch.equal(stopped, joint)
    torch.nn.Linear(7, 2)(joint).sum().backward()
    assert ca3.grad is not None and ca3.grad.abs().sum() > 0
    assert bypass.grad is not None and bypass.grad.abs().sum() > 0


def test_dg_gradient_probe_does_not_accumulate_and_reports_stop_vs_joint():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2, bias=False)

        def forward(self, value):
            return self.linear(value)

    features = torch.randn(6, 3)
    projection = Projection()
    decoder = torch.nn.Linear(2, 1, bias=False)
    dg = projection(features)
    encoder_loss = -dg.mean()
    stopped_loss = decoder(controller_core_view(dg, 2, "stop")).square().mean()
    stopped = dg_gradient_interaction_stats(stopped_loss, encoder_loss, projection)
    assert stopped["ppo_norm"].item() == 0.0
    assert stopped["encoder_norm"].item() > 0.0
    assert all(parameter.grad is None for parameter in projection.parameters())

    joint_loss = decoder(controller_core_view(dg, 2, "joint")).square().mean()
    joint = dg_gradient_interaction_stats(joint_loss, encoder_loss, projection)
    assert joint["ppo_norm"].item() > 0.0
    assert joint["encoder_norm"].item() > 0.0
    assert torch.isfinite(joint["cosine"])
    assert all(parameter.grad is None for parameter in projection.parameters())


def _contract_losses():
    projection = torch.nn.Linear(3, 2, bias=False)
    decoder = torch.nn.Linear(5, 1, bias=False)
    features = torch.randn(6, 3, requires_grad=True)
    dg = projection(features.detach())
    ca3 = 1.5 * dg
    controller_input = controller_core_view(torch.cat((ca3, features), dim=-1), 2)
    encoder_loss = -(dg * torch.tensor([1.0, 0.25])).mean()
    decoder_loss = decoder(controller_input).square().mean()
    return projection, decoder, features, encoder_loss, decoder_loss


def test_encoder_decoder_and_simultaneous_gradients_follow_graph_contract():
    projection, decoder, features, encoder_loss, decoder_loss = _contract_losses()
    parameters = (projection.weight, decoder.weight, features)
    enc = torch.autograd.grad(encoder_loss, parameters, retain_graph=True, allow_unused=True)
    dec = torch.autograd.grad(decoder_loss, parameters, retain_graph=True, allow_unused=True)
    combined = torch.autograd.grad(encoder_loss + decoder_loss, parameters, allow_unused=True)

    assert enc[0] is not None and enc[0].abs().sum() > 0
    assert enc[1] is None and enc[2] is None
    assert dec[0] is None or torch.equal(dec[0], torch.zeros_like(dec[0]))
    assert dec[1] is not None and dec[1].abs().sum() > 0
    assert dec[2] is not None and dec[2].abs().sum() > 0
    for enc_grad, dec_grad, total_grad, parameter in zip(enc, dec, combined, parameters):
        expected = torch.zeros_like(parameter)
        if enc_grad is not None:
            expected.add_(enc_grad)
        if dec_grad is not None:
            expected.add_(dec_grad)
        assert torch.allclose(total_grad, expected)


def test_running_consistent_batchnorm_updates_once_and_is_inference_consistent():
    projection = DGProjection_batchnorm_relu(
        3, 2, intercept=-10.0, batchnorm_semantics="running_consistent"
    )
    projection.train()
    x = torch.tensor([[1.0, 0.0, 2.0], [3.0, 2.0, 0.0], [5.0, 1.0, 4.0]])
    raw = projection.linear(x).detach()

    projection(x)
    assert int(projection.batchnorm1d.num_batches_tracked) == 0
    with projection.running_stats_update(True):
        learner_output = projection(x)
    assert int(projection.batchnorm1d.num_batches_tracked) == 1
    assert projection.last_running_stats_updated
    assert torch.allclose(projection.batchnorm1d.running_mean, raw.mean(dim=0))
    assert torch.allclose(projection.batchnorm1d.running_var, raw.var(dim=0, unbiased=False))

    with projection.running_stats_update(False):
        repeat_output = projection(x)
    assert int(projection.batchnorm1d.num_batches_tracked) == 1
    projection.eval()
    with projection.running_stats_update(True):
        actor_output = projection(x)
    assert int(projection.batchnorm1d.num_batches_tracked) == 1
    assert torch.allclose(learner_output, repeat_output)
    assert torch.allclose(repeat_output, actor_output)


def test_poststep_atomic_calibrates_updated_weights_and_publishes_one_generation():
    projection = DGProjection_batchnorm_relu(
        3, 2, intercept=-10.0, batchnorm_semantics="running_poststep_atomic"
    )
    projection.train()
    x = torch.tensor([[1.0, 0.0, 2.0], [3.0, 2.0, 0.0], [5.0, 1.0, 4.0]])

    with projection.running_stats_update(True):
        projection(x)
    assert int(projection.batchnorm1d.num_batches_tracked) == 0
    with torch.no_grad():
        projection.linear.weight.add_(0.2)
        normalize_dg_projection_rows(projection.linear)
        expected = projection.linear(x)
        assert projection.post_step_update_running_stats(torch.ones(3, dtype=torch.bool))

    assert int(projection.batchnorm1d.num_batches_tracked) == 1
    assert torch.allclose(projection.batchnorm1d.running_mean, expected.mean(dim=0))
    assert torch.allclose(
        projection.batchnorm1d.running_var, expected.var(dim=0, unbiased=False)
    )
    assert projection.weight_generation.item() == 1
    assert projection.statistics_generation.item() == 1

    projection.eval()
    actor_output = projection(x)
    projection.train()
    with projection.running_stats_update(False):
        decoder_only_output = projection(x)
    assert torch.allclose(actor_output, decoder_only_output)
    assert int(projection.batchnorm1d.num_batches_tracked) == 1


def test_input_centered_atomic_removes_feature_common_mode_before_projection():
    projection = DGProjection_batchnorm_relu(
        3, 2, intercept=-10.0, batchnorm_semantics="input_centered_atomic"
    )
    projection.train()
    x = torch.tensor([[10.0, 1.0, 3.0], [12.0, 4.0, 1.0], [14.0, 2.0, 5.0]])
    with projection.running_stats_update(True):
        projection(x)
    with torch.no_grad():
        normalize_dg_projection_rows(projection.linear)
        projection.post_step_update_running_stats(torch.ones(3, dtype=torch.bool))
        centered = x - x.mean(dim=0)
        expected_logits = projection.linear(centered)

    assert torch.allclose(projection.feature_running_mean, x.mean(dim=0))
    assert torch.allclose(projection.batchnorm1d.running_mean, expected_logits.mean(dim=0), atol=1e-6)
    assert projection.batchnorm1d.running_mean.abs().max() < 1e-5
    assert projection.feature_num_batches_tracked.item() == 1
    assert projection.weight_generation.item() == projection.statistics_generation.item() == 1


def test_generation_barrier_rejects_whole_mixed_rollouts_and_defers_small_fresh_batch():
    valids = torch.ones(4, 4, dtype=torch.bool)
    generation_matches = torch.tensor(
        [
            [True, True, True, True],
            [False, False, True, True],
            [True, True, True, True],
            [False, False, False, False],
        ]
    )
    accepted, stale = complete_rollout_generation_mask(valids, generation_matches)

    assert torch.equal(stale, torch.tensor([False, True, False, True]))
    assert accepted[0].all() and accepted[2].all()
    assert not accepted[1].any() and not accepted[3].any()
    assert require_sufficient_generation_batch(accepted, stale, minimum_valid_decisions=8)
    assert not require_sufficient_generation_batch(accepted, stale, minimum_valid_decisions=9)


def test_poststep_calibration_count_updates_returned_loss_summary_scope():
    summaries = {"additional_stats": {"dg_running_stats_update_count": torch.tensor(0.0)}}

    record_poststep_calibration_count(summaries, torch.tensor(2.0), calibrated=True)

    assert summaries["additional_stats"]["dg_running_stats_update_count"].item() == 1.0


def test_push_pull_rewards_have_opposite_distance_preferences():
    distances = torch.tensor([[0.0, 2.0, 2.0, 0.0], [0.0, 6.0, 6.0, 0.0]])
    decoder, encoder = legacy_reward_streams(distances, baseline=10.0, reward_scale=0.1, encoder_reward_method="encourage")
    assert encoder[1].mean() > encoder[0].mean()
    assert decoder[0].mean() > decoder[1].mean()


def test_replay_inactive_scheduled_row_has_zero_encoder_gradient():
    learner = SimpleNamespace(cfg=SimpleNamespace(Hippo_n_feature=2))
    head = torch.tensor([[1.0, 0.0]], requires_grad=True)
    scheduled = torch.tensor([[False, True]])
    applied = scheduled & head.detach().gt(0)
    loss = DistanceLearnerReward._encoder_loss(
        learner,
        head,
        torch.tensor([[0.0, 0.7]]),
        applied,
        torch.tensor([True]),
        0,
    )
    loss.backward()
    assert torch.equal(head.grad, torch.zeros_like(head))


def test_goal_adapter_reset_is_row_local_for_parameter_and_adam_state():
    parameter = torch.nn.Parameter(torch.arange(12.0).view(3, 4))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    parameter.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before_parameter = parameter.detach().clone()
    before_moments = {
        key: value.detach().clone()
        for key, value in optimizer.state[parameter].items()
        if torch.is_tensor(value) and value.shape == parameter.shape
    }
    learner = DistanceLearnerReward.__new__(DistanceLearnerReward)
    learner.actor_critic = SimpleNamespace(decoder=SimpleNamespace(target_modulation=parameter))
    learner.optimizer = optimizer
    learner._goal_adapter_reset_count = 0.0

    learner._reset_goal_adapter_row(1)

    assert torch.equal(parameter[1], torch.zeros_like(parameter[1]))
    assert torch.equal(parameter[0], before_parameter[0])
    assert torch.equal(parameter[2], before_parameter[2])
    for key, before in before_moments.items():
        after = optimizer.state[parameter][key]
        assert torch.equal(after[1], torch.zeros_like(after[1]))
        assert torch.equal(after[0], before[0])
        assert torch.equal(after[2], before[2])
    assert learner._goal_adapter_reset_count == 1.0


def test_generation_mask_rejects_only_stale_option_samples():
    layout = SimpleNamespace(persistent_start=2)
    graph = SimpleNamespace(representation_generation=torch.tensor(3))
    learner = SimpleNamespace(
        _uses_policy_graph=lambda: True,
        _hrl_state_from_rnn=lambda states: states,
        _hrl_layout=lambda: layout,
        _policy_graph=lambda: graph,
    )
    states = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 2.0]])
    matches = BaseDistanceRecorder._current_generation_mask(learner, states)
    assert torch.equal(matches, torch.tensor([True, False]))


def test_running_stats_and_film_state_round_trip_without_shape_change():
    projection = DGProjection_batchnorm_relu(3, 2, batchnorm_semantics="running_consistent")
    with projection.running_stats_update(True):
        projection(torch.randn(8, 3))
    core = SimpleNamespace(
        Hippo_n_feature=2,
        target_condition_start=5,
        get_out_size=lambda: 7,
    )
    film = TargetFiLMDecoder(core, hidden_size=4)
    with torch.no_grad():
        film.target_modulation[1].fill_(0.25)

    projection_clone = DGProjection_batchnorm_relu(3, 2, batchnorm_semantics="running_consistent")
    film_clone = TargetFiLMDecoder(core, hidden_size=4)
    projection_clone.load_state_dict(projection.state_dict())
    film_clone.load_state_dict(film.state_dict())

    assert torch.equal(projection_clone.batchnorm1d.running_mean, projection.batchnorm1d.running_mean)
    assert torch.equal(projection_clone.batchnorm1d.running_var, projection.batchnorm1d.running_var)
    assert torch.equal(film_clone.target_modulation, film.target_modulation)


def test_pre_atomic_checkpoint_remains_strictly_loadable():
    projection = DGProjection_batchnorm_relu(3, 2, batchnorm_semantics="running_poststep_atomic")
    legacy_state = projection.state_dict()
    for name in (
        "feature_running_mean",
        "feature_num_batches_tracked",
        "weight_generation",
        "statistics_generation",
    ):
        legacy_state.pop(name)

    restored = DGProjection_batchnorm_relu(3, 2, batchnorm_semantics="running_poststep_atomic")
    restored.load_state_dict(legacy_state, strict=True)

    assert restored.feature_num_batches_tracked.item() == 0
    assert restored.weight_generation.item() == 0
    assert restored.statistics_generation.item() == 0
