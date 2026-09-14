import gymnasium as gym
import numpy as np
import torch

from hpc_runs.intrmotiv_study import load_study
from sample_factory.model.actor_critic import create_actor_critic
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import parse_dmlab_args, register_dmlab_components

register_dmlab_components()
torch.set_num_threads(1)
run = next(
    r
    for r in load_study("hpc_runs/studies/dg_capacity_goal_conditioning_preflight.study.json").expand_runs()
    if r.base == "DIRECT_DG"
)
cfg = parse_dmlab_args(argv=list(run.args) + ["--experiment=dg_capacity_model_smoke"])
obs_space = gym.spaces.Dict(
    {
        "obs": gym.spaces.Box(0, 255, (4, 72, 96), np.uint8),
        "INSTR": gym.spaces.Box(1, 3, (1,), np.int32),
        "telemetry_pose": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
    }
)
m = create_actor_critic(cfg, obs_space, gym.spaces.Discrete(8))
m.train()
x = {
    "obs": torch.rand(8, 4, 72, 96) * 255,
    "INSTR": torch.ones(8, 1, dtype=torch.long),
    "telemetry_pose": torch.zeros(8, 4),
}
h = m.forward_head(x)
s = torch.zeros(8, cfg.rnn_size)
goal = torch.nn.functional.one_hot(torch.arange(8) % cfg.Hippo_n_feature, cfg.Hippo_n_feature).float()
# Provide above-threshold synthetic logits to ensure active modulation gradients,
# while still exercising the actual encoder's output layout and normalization.
h = h.clone()
h[:, -cfg.Hippo_n_feature :] = 3
replay = torch.cat((h, goal), -1)
o, ns = m.forward_core(replay, s)
y = m.forward_tail(o, values_only=False, sample_actions=False)
loss = y["action_logits"].square().mean() + y["values"].square().mean()
loss.backward()
grad = m.core.dg_goal_modulation.grad.norm().item()
assert grad > 0
print("MODEL_SMOKE_PASS", h.shape, o.shape, ns.shape, "gradient", grad)
