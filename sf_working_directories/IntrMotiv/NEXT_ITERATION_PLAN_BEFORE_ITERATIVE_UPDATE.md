# Next HRL Iteration Plan (Preserved Before Iterative-Update Work)

This file preserves the planned HRL next iteration while implementation work first focuses on the encoder/decoder iterative update. It is deferred, not cancelled.

## Objective

Run a convincing controllable-graph HRL experiment long enough to distinguish failure to learn from insufficient training. A four-policy PBT population is the scientific unit and should receive approximately 100M environment steps in total; report the population as one PBT replicate, not as four independent replicates.

## Fixed architecture

- Use the local `IntrMotiv` implementation and remain compatible with the baseline Jannek Sample Factory workflow.
- Use `layer2_resnet18`, not a pretrained ResNet.
- Keep the DG representation as the subgoal representation, with `F=16` and `L=64` for the next focused run unless the iterative-update validation changes that decision.
- Keep the existing intrinsic-reward signals, including both `hit` and `hit_distance`; do not remove hit distance.
- Keep `encoder_batch_loss` enabled and monitor DG density, multi-activation rate, target-hit rate, and silent-unit fraction.
- Do not make normalization a prerequisite for this iteration.

## HRL and planning changes

- Use the episode-local controllable graph stored in recurrent state for PPO consistency.
- Keep target selection exploration-driven. Cheapness is not the objective and should not be a major sweep axis.
- Add feasibility-aware planning: prefer currently reachable targets, and use a multi-hop/frontier fallback when most subgoals are farther than `L` steps.
- Let the option horizon be experience-derived: when the expected time is nearly exhausted without reaching the target, switch to another target within a small margin, following the “World Models as a Graph” idea.
- Preserve target/source ids, option resets, target hits, controllability updates, and hit-distance diagnostics.

## Predictor extension

Evaluate the CA3-based predictor as a shadow/auxiliary diagnostic first. The predictor should map the entire CA3 state `S` (not only past DG id and age) to a predictive next DG/F state. Compare it with the controllable reachability matrix: the predictor estimates local temporal successor structure, while the reachability matrix stores successful controllable source-to-target experience. Do not replace the matrix until the distinction is measured.

## Batch and accounting

- Use four policies per PBT population so experience and parameters carry over.
- Average metrics across the four policies for each PBT population; treat their summed steps as the approximately 100M population budget, not as four independent 100M runs.
- Retain the existing Slurm resource conventions unless a measured bottleneck justifies a change.
- Omit an artificial Sample Factory training-frame cap; use the 100M population-level target so runs cannot silently stop near a few million steps.
- Sample Factory applies `train_for_env_steps` per policy and stops when all four policies cross it. Set the production CLI target to `25M` per policy, yielding approximately `100M` aggregate steps for the four-policy population.
- Request roughly 30 hours per allocation. This is below the NEMO2 `cpu` partition's four-day maximum and should improve scheduling/concurrency while leaving margin above the expected 20-40 hour runtime. If a run needs another allocation, resubmit the same description and resume from checkpoints.
- Use a current W&B SDK (at least `0.25.0`) to avoid the known old TensorBoard upload issue.
- Keep batch templates, submission scripts, logs, and errors inside `IntrMotiv` where possible, preserving compatibility with the old workflow.

## Suggested first sweep

Use a small, interpretable batch rather than sweeping many coupled knobs. Sweep
the three encoder reward methods (`punish`, `encourage`, `mean`) and the two
worker reward modes (`hit`, `hit_distance`) with three seeds, giving 18 jobs.
Keep the iterative-update schedule fixed during this batch so its effect is
not confounded with HRL architecture changes. Compare population-level curves
and diagnostics, not isolated policy curves.

## Success and failure criteria

Success requires sustained DG activity, nonzero target hits and low-level intrinsic reward, increasing controllable-graph coverage, and navigation success that continues improving through the 100M population budget. If activity remains nonzero but target hits remain near zero, prioritize feasibility/planning and option timing. If DG activity collapses, stop the HRL interpretation and debug encoder rewards/gradient ownership first.
