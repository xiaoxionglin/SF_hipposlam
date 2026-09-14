# Topological manager reference fixture

`topological_manager_reference.npz` contains nine complete, deterministic input
cases and five successive expected transitions per case. Generated on 2026-09-15
using the unmodified `dmlab/topological_frontier.py` from runtime commit
`d94155f9be0436828ee9a744b57097db07022344` (file SHA-256
`fb750827843e5db5f91779a4ceb387b4c7f2ed777819dae5349fb0d960e5e8c7`).
PyTorch 2.7.1, CPU, one thread. Expected outputs were checked against the batched
implementation before saving. Do not regenerate expected outputs from the new
implementation when testing equivalence.

Each case stores its keyword arguments as JSON text, all graph buffers, the
initial option/topology tensors, and each step's DG activity, action features,
expected option state, expected topology state, and expected policy condition.
Arrays load without pickle. Inputs are stored explicitly so future tests do not
depend on random-number implementation details.

The cases sample frontier/least-tested/local-successor selection, direct/waypoint
planning, ordinary/common manager, no geometry/SE(2), immediate/delayed timing,
motion filtering, and target-hit/first-distinct outcomes. They include stale
generations, return/validation state, silent and multiply active DG observations.
Every tensor and graph buffer is compared; tests also check that inputs and the
graph remain unchanged. CUDA repeats the same oracle comparisons when available.

The one-time broader comparison covered 192 parameter combinations, each with
24 streams, seven nodes, and five successive transitions. Another nine cases
verified shared bookkeeping under the existing edge-probing planner.
