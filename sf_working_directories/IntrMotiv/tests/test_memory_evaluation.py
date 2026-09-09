import numpy as np
import pytest
import pandas as pd
import torch
from types import SimpleNamespace
from sf_working_directories.IntrMotiv.evaluation.absent_goal_interventions import summarize_commands
from sf_working_directories.IntrMotiv.evaluation.observation_panel import save_panel
from sf_working_directories.IntrMotiv.evaluation.qualify_absent_goal_interventions import canonical_components, qualify
from sf_working_directories.IntrMotiv.evaluation.ca3_memory_behavior import behavior_diagnostics


def test_command_control_detects_command_specificity_not_unconditional_visits():
    rows = [dict(start=0, command=0, complete=True, eligible_count=2, first_onset_times=[8, 0]),
            dict(start=0, command=1, complete=True, eligible_count=2, first_onset_times=[0, 8])]
    assert summarize_commands(rows, 2)['ctc_auc'] == 1
    rows[1]['first_onset_times'] = [8, 0]
    assert summarize_commands(rows, 2)['ctc_auc'] == 0
    rows[1]['complete'] = False
    assert summarize_commands(rows, 2)['ctc_auc'] is None


def test_observation_panel_rejects_home_storage(tmp_path):
    with pytest.raises(ValueError):
        save_panel(tmp_path / 'panel.npz', {'dones': [np.zeros(1)]})


def test_common_panel_records_actions_for_contextual_replay_without_policy_input_changes():
    from collections import defaultdict
    from sf_working_directories.IntrMotiv.evaluation.observation_panel import record_observation, previous_actions_after_step
    obs={'obs':torch.ones(2,3)};records=defaultdict(list)
    previous=np.full((2,1),6,np.int32)
    record_observation(records,obs,previous)
    assert 'prev_action' not in obs
    assert records['obs_prev_action'][0].tolist()==[[6],[6]]
    previous=previous_actions_after_step([[2],[4]],[False,True],6)
    record_observation(records,obs,previous)
    assert records['obs_prev_action'][1].tolist()==[[2],[6]]
    assert records['obs_prev_action'][0].tolist()==[[6],[6]]


def test_spatial_qualification_allows_later_canonical_arrival_and_excludes_secondary_field():
    maps = np.zeros((3, 3, 1))
    maps[0, 0, 0], maps[2, 2, 0] = 2, 1.1
    masks = canonical_components(maps, np.ones((3, 3)))
    assert masks.sum() == 1 and masks[0, 0, 0]
    rows = [dict(events=[[1, 0, 1800, 1800], [4, 0, 150, 150]])]
    assert qualify(rows, masks)[0]['first_onset_times'] == [4]


def test_behavior_repetition_respects_episode_boundaries():
    pose = pd.DataFrame(dict(agent=[0]*8, num_traj=[0]*4+[1]*4, x=np.arange(8), y=[0]*8))
    dg = np.tile(np.eye(2), (4, 1))
    result = behavior_diagnostics(pose, dg, refractory=0, window=2)
    assert result['dominant_onsets'] == 8
    assert result['repeated_motif_period_2_fraction'] == 1
    assert result['repeated_motif_period_4_fraction'] is None
    assert result['window_straightness_p10_p50_p90'] == [1, 1, 1]


def test_panel_replay_preserves_pose_and_episode_resets(tmp_path, monkeypatch):
    from sf_working_directories.IntrMotiv.evaluation import observation_panel as panel
    monkeypatch.setattr(panel, 'get_rnn_size', lambda cfg: 4)
    monkeypatch.setattr(panel, 'prepare_and_normalize_obs', lambda actor, obs: obs)
    class Actor:
        encoder = SimpleNamespace(DG_projection=SimpleNamespace(last_pre_threshold_logits=None))
        def forward_head(self, obs):
            self.encoder.DG_projection.last_pre_threshold_logits = obs['obs'].float()
            return obs['obs'].float()
        def forward_core(self, head, state):
            output = state + head.repeat_interleave(2, dim=-1)
            return output, output
    path = tmp_path / 'panel.npz'
    np.savez_compressed(path, obs_obs=np.ones((3,1,2)), obs_pos=np.arange(9).reshape(3,1,3),
                        obs_rot=np.zeros((3,1,3)), dones=np.array([[False],[True],[False]]))
    pose, activity, _ = panel.replay_observations(Actor(), SimpleNamespace(Hippo_n_feature=2, Hippo_R=1, Hippo_L=2), path, 'cpu')
    assert activity[:, 0].tolist() == [1,2,1]
    assert pose.num_traj.tolist() == [0,0,1]
    assert pose.x.tolist() == [0,3,6]
