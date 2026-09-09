"""CPU tests exercise the actual SF learner/core; no DMLab installation needed."""

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from sample_factory.algo.learning.learner import DefaultLearner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_context import global_model_factory, reset_global_model_context
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.model.encoder import Encoder
from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.default.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.default.dmlab.custom_params import add_hipposlam_env_args
from sf_xxl.dmlab.world_model import DGEventWorldModel, event_prediction_loss, next_dg_event_labels
from sf_xxl.dmlab.world_model_learner import DGWorldModelLearner, add_world_model_args, make_world_model_learner


@pytest.fixture(autouse=True)
def isolated_model_factory():
    reset_global_model_context()
    yield
    reset_global_model_context()


def test_first_future_vector_preserves_simultaneous_and_continuing_activity():
    dg = torch.tensor([[[1., 0.], [1., 2.], [0., 0.], [0., 3.], [0., 0.]]])
    labels = next_dg_event_labels(dg, torch.zeros(1, 5, dtype=torch.bool), torch.ones(1, 5, dtype=torch.bool), 3)
    assert labels.delay.tolist() == [[1., 2., 1., 0., 0.]]
    assert labels.dg[0, 0].tolist() == [1., 2.]
    assert labels.usable.tolist() == [[True, True, True, False, False]]


def test_reset_padding_and_recurrence_tail_are_censored():
    dg = torch.zeros(2, 5, 2)
    dg[0, 2] = 1  # reset observation must not label the preceding episode
    dones = torch.zeros(2, 5, dtype=torch.bool)
    dones[0, 1] = True
    valid = torch.ones_like(dones)
    valid[1, 1] = False
    labels = next_dg_event_labels(dg, dones, valid, 2)
    assert not labels.hit[0, :2].any()
    assert not labels.usable[0, :2].any()
    assert not labels.usable[1, :2].any()
    assert labels.usable[1, 2]
    assert not labels.usable[:, 3:].any()
    assert not next_dg_event_labels(torch.zeros_like(dg), dones, valid, 8).usable.any()


def test_model_uses_candidate_action_without_mutating_state():
    model = DGEventWorldModel(4, 2, 3, 8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[0].weight[0, 4] = 1  # action zero only
        model.network[2].weight[0, 0] = 1
    ca3 = torch.zeros(2, 4)
    before = ca3.clone()
    result = model.predict_actions(ca3, 8)
    assert result["hit_probability"].shape == (2, 3)
    assert result["dg_probability"].shape == (2, 3, 2)
    assert (result["hit_probability"][:, 0] > result["hit_probability"][:, 1]).all()
    assert torch.equal(ca3, before)


def test_event_loss_only_updates_predictor_and_handles_no_labels():
    model = DGEventWorldModel(4, 2, 3, 8)
    ca3 = torch.randn(4, 4, requires_grad=True)
    dg = torch.tensor([[[0., 0.], [1., 2.], [0., 0.], [0., 0.]]], requires_grad=True)
    valid = torch.ones(1, 4, dtype=torch.bool)
    labels = next_dg_event_labels(dg, torch.zeros_like(valid), valid, 2)
    loss, _ = event_prediction_loss(model, ca3, torch.zeros(4, dtype=torch.long), labels, 2)
    loss.backward()
    assert ca3.grad is None and dg.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    empty = next_dg_event_labels(torch.zeros_like(dg), torch.zeros_like(valid), valid, 8)
    zero, _ = event_prediction_loss(model, ca3, torch.zeros(4, dtype=torch.long), empty, 8)
    assert zero.item() == 0


class ToyDGEncoder(Encoder):
    """A trainable DG-like encoder for exercising the genuine SF PPO update."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.projection = torch.nn.Linear(3, 2)

    def forward(self, obs):
        return self.projection(obs["obs"]).relu()

    def get_out_size(self):
        return 2


def make_config(tmp_path, enabled):
    argv = ["--env=world_model_unit_test", "--experiment=" + ("on" if enabled else "off"),
            "--train_dir=" + str(tmp_path), "--device=cpu", "--serial_mode=true", "--seed=12",
            "--dg_world_model=" + str(enabled), "--dg_world_model_horizon=2", "--dg_world_model_hidden_size=8",
            "--core_name=BypassSS", "--Hippo_n_feature=2", "--Hippo_R=2", "--Hippo_L=3",
            "--use_rnn=true", "--rnn_type=gru", "--rnn_size=8", "--recurrence=4", "--rollout=4",
            "--batch_size=8", "--num_batches_per_epoch=1", "--num_epochs=1",
            "--normalize_input=false", "--normalize_returns=false", "--decoder_mlp_layers=8",
            "--learning_rate=0.001", "--lr_schedule=constant"]
    parser, _ = parse_sf_args(argv)
    add_hipposlam_env_args(parser)
    add_world_model_args(parser)
    return parse_full_cfg(parser, argv)


def make_learner(tmp_path, enabled):
    reset_global_model_context()
    global_model_factory().register_encoder_factory(ToyDGEncoder)
    global_model_factory().register_model_core_factory(SimpleSequenceWithBypassCore)
    cfg = make_config(tmp_path, enabled)
    info = EnvInfo(gym.spaces.Dict(obs=gym.spaces.Box(-1, 1, (3,), dtype=np.float32)),
                   gym.spaces.Discrete(3), 1, False, False, None, None, 1)
    versions = torch.zeros(1, dtype=torch.int32)
    server = ParameterServer(0, versions, serial_mode=True)
    learner = make_world_model_learner(cfg, info, versions, 0, server)
    learner.init()
    return learner


def minibatch(learner):
    torch.manual_seed(23)
    n = 8
    batch = TensorDict(
        normalized_obs=TensorDict(obs=torch.randn(n, 3)), rnn_states=torch.zeros(n, 8),
        actions=torch.tensor([[0], [1], [2], [0], [1], [0], [2], [1]]),
        dones=torch.zeros(n, dtype=torch.bool), dones_cpu=torch.zeros(n, dtype=torch.bool),
        valids=torch.ones(n, dtype=torch.bool), advantages=torch.linspace(-1, 1, n),
        returns=torch.linspace(0, 1, n), values=torch.zeros(n),
        policy_id=torch.zeros(n, dtype=torch.long), policy_version=torch.zeros(n),
    )
    # Include an episode boundary, exercising SF's packed replay and reset path.
    batch["dones"][1] = batch["dones_cpu"][1] = True
    with torch.no_grad():
        outputs = learner._forward_pass(AttrDict(batch), 4, batch["valids"])
        batch["action_logits"] = outputs.result["action_logits"].detach().clone()
        batch["log_prob_actions"] = learner.actor_critic.action_distribution().log_prob(batch["actions"]).detach()
    return batch


def assert_same_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_real_ppo_training_matches_flag_off_and_restores_world_model_checkpoint(tmp_path):
    baseline = make_learner(tmp_path, False)
    assert type(baseline) is DefaultLearner
    state = copy.deepcopy(baseline.actor_critic.state_dict())
    baseline_rng = torch.get_rng_state().clone()
    batch = minibatch(baseline)
    enabled = make_learner(tmp_path, True)
    assert isinstance(enabled, DGWorldModelLearner)
    assert torch.equal(baseline_rng, torch.get_rng_state())
    assert_same_state(state, enabled.actor_critic.state_dict())
    assert not any("world_model" in key for key in enabled.actor_critic.state_dict())
    head_before = copy.deepcopy(enabled.world_model.state_dict())
    # Run the real inherited PPO training loop, including the post-optimizer hook
    # under no_grad, gradient clipping, summaries, and parameter-server updates.
    baseline._should_save_summaries = enabled._should_save_summaries = lambda: True
    baseline._train(copy.deepcopy(batch), 8, 8, 0)
    stats = enabled._train(copy.deepcopy(batch), 8, 8, 0)
    assert_same_state(baseline.actor_critic.state_dict(), enabled.actor_critic.state_dict())
    assert any(not torch.equal(value, enabled.world_model.state_dict()[key]) for key, value in head_before.items())
    assert stats["dg_world_model/usable_count"] > 0
    assert enabled._world_model_pending is None
    checkpoint = copy.deepcopy(enabled._get_checkpoint_dict())
    assert checkpoint["dg_world_model"]["optimizer"]["state"]
    assert enabled.save()
    restored = make_learner(tmp_path, True)  # actual startup loads saved checkpoint
    assert_same_state(enabled.world_model.state_dict(), restored.world_model.state_dict())
    assert_same_state(enabled.actor_critic.state_dict(), restored.actor_critic.state_dict())
    # Next identical step verifies optimizer moments, not just parameter loading.
    enabled._train(copy.deepcopy(batch), 8, 8, 0)
    restored._train(copy.deepcopy(batch), 8, 8, 0)
    assert_same_state(enabled.world_model.state_dict(), restored.world_model.state_dict())
    # Legacy checkpoint loading leaves the PPO schema unchanged and resets only head.
    restored._load_state(copy.deepcopy(baseline._get_checkpoint_dict()))
    assert_same_state(baseline.actor_critic.state_dict(), restored.actor_critic.state_dict())
    assert not restored.world_model_optimizer.state


def test_censored_only_update_does_not_advance_optimizer_or_mutate_actor(tmp_path):
    learner = make_learner(tmp_path, True)
    before = copy.deepcopy(learner.actor_critic.state_dict())
    head_before = copy.deepcopy(learner.world_model.state_dict())
    # Each row invalid, with out-of-range action sentinel. It must not supervise.
    learner._world_model_pending = (torch.zeros(4, 8), torch.zeros(4, 2),
                                    torch.full((4, 1), -1), torch.zeros(4, dtype=torch.bool),
                                    torch.zeros(4, dtype=torch.bool), 4)
    with torch.no_grad():
        learner._after_optimizer_step()
    assert_same_state(before, learner.actor_critic.state_dict())
    assert_same_state(head_before, learner.world_model.state_dict())
    assert not learner.world_model_optimizer.state
    assert learner._world_model_stats["dg_world_model/usable_count"] == 0


def test_invalid_configuration_fails_early(tmp_path):
    learner = make_learner(tmp_path, False)
    cfg = make_config(tmp_path, True)
    cfg.recurrence = 1
    with pytest.raises(ValueError, match="recurrence"):
        make_world_model_learner(cfg, learner.env_info, learner.policy_versions_tensor, 0, learner.param_server)
