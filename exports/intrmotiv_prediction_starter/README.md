# IntrMotiv prediction starter

A small handoff for the **action input | planning | world model** project.
Start with prediction as a separate, measurable module. The original actor
continues selecting every real action. This kit does not implement a planner
or claim improved transfer performance.

## Run first

Use a Python environment with PyTorch and pytest. From this directory:

```bash
python -m pytest -q
python -m examples.shadow_training
```

No Sample Factory, DMLab, GPU, checkpoint, network connection, or editable
installation is needed for those commands. The supplied code requires Python
3.10 or newer. Local verification versions are recorded in VALIDATION.md.
If integrating as a package into an already provisioned environment, use
`python -m pip install -e . --no-deps --no-build-isolation` (requires setuptools).

## What to open

| File | Purpose | Status |
|---|---|---|
| `intrmotiv_transfer/shadow.py` | CA3 + specified target -> hit logit and conditional time | Head extracted from IntrMotiv |
| `intrmotiv_transfer/contextual_dg.py` | Next distinct landmark/timeout classifier; optional contextual DG feedback | Original standalone source, unchanged |
| `intrmotiv_transfer/labels.py` | Future target labels, with incomplete windows explicitly censored | New corrected handoff adapter |
| `intrmotiv_transfer/losses.py` | Detached, head-only training loss | Adapted from IntrMotiv shadow loss |
| `intrmotiv_transfer/state.py` | Causal action history; pure CA3 step and silent skip | New standalone adapters of existing conventions |
| `INTEGRATION.md` | Where these tensors enter a Sample Factory learner | Integration contract, not a drop-in patch |
| `STUDENT_TASK.md` | First experiment and remaining world-model work | Project boundary |

## The first milestone

**Attach a shadow predictor to the student's existing model, without changing
the policy's inputs, action selection, or real recurrent-state updates.**
For the supplied head, a target is only a prediction query. It is not a
goal-conditioned control command and need not be fed to the actor.

Measure useful event prediction against a simple target-frequency baseline on
held-out real trajectories. The toy example only checks wiring and gradients.
Run the baseline with the exact same backbone, DG detector, and controller as
before; do not replace them with IntrMotiv's experimental variants.

## Important boundaries

- The exported heads are **not action-conditioned world models**. Add future
  candidate action conditioning explicitly in the student's project.
- `prev_action` means the executed action that produced the current
  observation; it is not the next candidate action.
- The shadow head predicts occurrence of a specified target within a horizon.
  It does not predict which DG event happens first.
- The other head predicts a **first distinct exclusive landmark or timeout**
  under the IntrMotiv option contract. This does not mean intervening DG
  activity is zero. It predicts identity, not the full DG amplitudes/vector.
- Silent skipping is exact only when every omitted DG vector is zero. It
  updates the CA3 register alone, not bypass observations or action histories.
- The original contextual predictor propagates gradients into source DG.
  For predictor-only work detach inputs. Contextual DG feedback is optional
  reference code; leave it disabled for the initial experiment.
- The old shadow label helper was not copied: it treated unfinished horizons
  as negatives. The supplied replacement masks them as censored. The original
  option-batch extractor still drops recurrence-crossing source events; it is
  reference code, not the recommended long-horizon data collector.
- Clipped class-weighted BCE helps training but its hit probabilities are not
  automatically calibrated. Validate calibration before using them as probabilities.
- No graph manager, intrinsic reward, recruitment, goal-conditioned worker,
  imagination-to-policy loss, or persistent replay database is included.

## Provenance

See PROVENANCE.json for source-file hashes and extraction/adaptation details.
Source was inspected on NEMO2 on 2026-09-10, repository HEAD `636be3a5`, with
working-tree modifications. The complete learner/core is deliberately omitted
to avoid importing unrelated research mechanisms. No third-party dependency
source or checkpoints are bundled; no new license is asserted by this handoff.
