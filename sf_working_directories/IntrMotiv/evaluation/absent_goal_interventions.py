"""Graph-free command interventions with exactly replayed starting histories.

All eligible absent commands are executed from each start. Other commands form
a within-start, leave-one-command-out control, not a post-hoc label shuffle.
"""
import json
import numpy as np
import pandas as pd
import torch
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

HORIZONS = (8, 16, 32, 64)


def summarize_commands(rows, n_nodes, horizons=HORIZONS):
    """Macro-average identity-specific matched-minus-other-command hit rates."""
    results = {}
    for horizon in horizons:
        identity_deltas = []
        for target in range(n_nodes):
            differences = []
            for start in sorted({row['start'] for row in rows}):
                trials = [r for r in rows if r['start'] == start and r['complete']]
                matched = [r for r in trials if r['command'] == target]
                other = [r for r in trials if r['command'] != target]
                # Incomplete command sets must not become survivorship controls.
                if not matched or not other or len(trials) != matched[0]['eligible_count']:
                    continue
                hit = lambda r: 0 < r['first_onset_times'][target] <= horizon
                differences.append(float(hit(matched[0])) - np.mean([hit(r) for r in other]))
            if differences:
                identity_deltas.append(float(np.mean(differences)))
        results[f'ctc_{horizon}'] = float(np.mean(identity_deltas)) if identity_deltas else None
        results[f'ctc_{horizon}_represented_identities'] = len(identity_deltas)
    values = [results[f'ctc_{h}'] for h in horizons]
    results['ctc_auc'] = float(np.trapz(values, horizons) / (horizons[-1] - horizons[0])) if len(horizons) > 1 and all(v is not None for v in values) else None
    results['spatially_qualified_ctc'] = None
    results['spatial_qualification_status'] = 'Requires independent canonical-field maps; identity CTC alone is not spatial control.'
    return results


def run_absent_goal_interventions(cfg, env, env_info, actor, checkpoint, device,
                                  decision_cap, deterministic=False, starts=16, prefix_length=128,
                                  max_commands=None, horizons=HORIZONS):
    horizons = tuple(int(h) for h in horizons)
    if not horizons or sorted(set(horizons)) != list(horizons) or horizons[0] <= 0:
        raise ValueError('Horizons must be positive, unique, and increasing')
    horizon = horizons[-1]
    if env.num_agents != 1:
        raise ValueError('Matched reset interventions require one environment')
    n, e, refractory = int(cfg.Hippo_n_feature), int(cfg.Hippo_R + cfg.Hippo_L - 1), int(cfg.Hippo_R)
    if getattr(actor.core, 'policy_graph', None) is not None:
        raise ValueError('Absent-goal interventions must be graph-free')
    base_env = env.unwrapped
    if not callable(getattr(base_env, 'seed', None)):
        raise ValueError('Environment does not expose deterministic seed/reset')
    # Cached levels consume an independent seed list; bypass that for replay.
    if hasattr(base_env, 'curr_cache'):
        base_env.curr_cache = None
        base_env.use_level_cache = False
    parameters_before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    rows, decisions, skipped = [], 0, 0

    def encode(obs, state):
        head = actor.forward_head(prepare_and_normalize_obs(actor, obs))
        return actor.forward_core(head, state)

    def step(action):
        nonlocal decisions
        if decisions >= decision_cap:
            raise RuntimeError('Intervention decision cap exhausted')
        action = torch.as_tensor([[action]], device=device)
        result = env.step(preprocess_actions(env_info, action))
        decisions += 1
        return result[0], bool(make_dones(result[2], result[3])[0])

    def reset_prefix(start):
        nonlocal env, base_env
        # DMLab can retain engine state across reset(seed=...). Reconstruct the
        # environment for each branch; a mere reseed is not a matched start.
        env.close()
        evaluation_cfg = AttrDict(dict(cfg))
        evaluation_cfg.dmlab_use_level_cache = False
        env = make_env_func_batched(evaluation_cfg,
            env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None)
        base_env = env.unwrapped
        if hasattr(base_env, 'reset_on_init'):
            base_env.reset_on_init = False
        seed = 10000 + start
        base_env.seed(seed)
        torch.manual_seed(seed)
        obs, _ = env.reset()
        state = torch.zeros(1, get_rnn_size(cfg), device=device)
        actions = np.random.default_rng(seed).integers(0, actor.action_space.n, prefix_length)
        for action in actions:
            _, state = encode(obs, state)
            obs, done = step(int(action))
            if done:
                raise RuntimeError('Episode boundary in deterministic intervention prefix')
        output, state = encode(obs, state)
        signature = {k: v.detach().clone() for k, v in obs.items() if torch.is_tensor(v)}
        return obs, output, state, signature

    try:
        with torch.no_grad():
            for start in range(starts):
                # Reserve worst-case cost before choosing a start; never select by success.
                if decisions + prefix_length + (max_commands or n) * (prefix_length + horizon) > decision_cap:
                    break
                _, reference_output, reference_state, signature = reset_prefix(start)
                ca3 = reference_state[:, :n * e].reshape(1, n, e)
                eligible = torch.nonzero(ca3.amax(-1)[0].eq(0)).flatten().tolist()
                if max_commands is not None:
                    eligible = eligible[:max_commands]
                if len(eligible) < 2:
                    skipped += 1
                    continue
                for command in eligible:
                    obs, output, state, replay_signature = reset_prefix(start)
                    if signature.keys() != replay_signature.keys() or any(
                        not torch.equal(value, replay_signature[key]) for key, value in signature.items()
                    ) or not torch.equal(reference_state, state) or not torch.equal(reference_output, output):
                        differences = {key: float((value.float() - replay_signature[key].float()).abs().max())
                                       for key, value in signature.items() if key in replay_signature}
                        differences['state'] = float((reference_state - state).abs().max())
                        differences['core_output'] = float((reference_output - output).abs().max())
                        raise RuntimeError(f'Start {start} is not exactly reproducible: {differences}; cannot claim matched intervention')
                    times, positions, trajectory, events = [0] * n, [None] * n, [], []
                    rng = torch.Generator(device=device).manual_seed(20000 + start)
                    complete, path_length = True, 0.0
                    last_pos = obs['pos'][0, :2].cpu().numpy().copy()
                    for elapsed in range(1, horizon + 1):
                        conditioned = output.clone()
                        offset = int(actor.core.target_condition_start)
                        conditioned[:, offset:offset + n] = 0
                        conditioned[:, offset + command] = 1
                        result = actor.forward_tail(conditioned, values_only=False, sample_actions=False)
                        probabilities = result['action_logits'].softmax(-1)
                        action = probabilities.argmax(-1) if deterministic else torch.multinomial(probabilities, 1, generator=rng)
                        previous = state[:, :n * e].reshape(1, n, e).clone()
                        obs, done = step(int(action.item()))
                        if done:
                            complete = False
                            break
                        output, state = encode(obs, state)
                        current = state[:, :n * e].reshape(1, n, e)[..., 0]
                        candidates = current.gt(0) & ~previous[..., :refractory].gt(0).any(-1)
                        pos = obs['pos'][0, :2].cpu().numpy().copy()
                        path_length += float(np.linalg.norm(pos - last_pos))
                        trajectory.append(pos.tolist())
                        last_pos = pos
                        if candidates.any():
                            winner = int(current.masked_fill(~candidates, -torch.inf).argmax(-1))
                            events.append([elapsed, winner, *pos.tolist()])
                            if times[winner] == 0:
                                times[winner], positions[winner] = elapsed, pos.tolist()
                    rows.append(dict(start=start, command=command, eligible_count=len(eligible),
                        eligible_commands=eligible, complete=complete, first_onset_times=times,
                        first_onset_positions=positions, trajectory=trajectory, events=events, path_length=path_length,
                        start_position=signature['pos'][0].cpu().tolist(),
                        start_rotation=signature['rot'][0].cpu().tolist()))
    finally:
        env.close()
    if any(not torch.equal(v, actor.state_dict()[k]) for k, v in parameters_before.items()):
        raise RuntimeError('Policy or normalization buffers changed during frozen intervention')
    summary = dict(protocol='absent-goal-matched-commands-v1', checkpoint=str(checkpoint),
        decisions=decisions, decision_cap=decision_cap, completed_trials=sum(r['complete'] for r in rows),
        starts_with_insufficient_absent_targets=skipped, starts_requested=starts,
        starts_evaluated=len({r['start'] for r in rows}), prefix_length=prefix_length,
        max_commands=max_commands,
        exact_start_verified=True, policy_frozen=True, dg_frozen=True,
        horizons=list(horizons),
        termination=f'{horizon} decisions or episode boundary; intermediate landmarks do not terminate',
        **summarize_commands(rows, n, horizons))
    frame = pd.DataFrame(rows)
    for column in ('eligible_commands', 'first_onset_times', 'first_onset_positions', 'trajectory', 'events', 'start_position', 'start_rotation'):
        if column in frame:
            frame[column] = frame[column].map(json.dumps)
    return frame, summary
