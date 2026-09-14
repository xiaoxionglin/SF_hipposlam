# Verification

Date: 2026-09-10. CPU, Python 3.10.12, PyTorch 2.7.1, pytest 9.0.2.

Commands run from the starter directory:

```bash
python -m pytest -q
python -m examples.shadow_training
```

Results:

- **21 tests passed.** Includes seven extracted standalone IntrMotiv tests and
  fourteen new test cases for the handoff contracts (counting parametrization).
- Synthetic head training loss decreased from **0.9685 to 0.0147**. Actor
  outputs remained exactly unchanged; no gradient reached the representation.
- Tests cover episode/reset and padding censorship, incomplete horizons,
  positive-only time error, action history, head-only gradient isolation, and
  exact agreement of silent CA3 skipping with explicit stepwise evolution.
- A negative-control test confirms that skipping a nonzero intermediate DG
  input does not reproduce the true CA3 state.
- Extracted contextual source is byte-identical to the audited source, and
  the shadow head's class body is identical. See PROVENANCE.json.

Limits:

- No integration into the student's repository has been attempted.
- No DMLab, Sample Factory rollout, GPU, or real navigation/transfer evaluation
  was run for this bundle. The synthetic loss reduction is a wiring check only.
- Full IntrMotiv packed-replay tests depend on its complete runtime and are
  not bundled as standalone tests. The student must implement the corresponding
  integration checks listed in INTEGRATION.md in her repository.
- The old upstream incomplete-window labeling behavior is corrected only in
  this handoff adapter; the NEMO2 runtime was not modified.

Preparation lesson: standalone extraction initially missed the shadow class's
`torch` import; the runnable tests caught it and it was fixed before packaging.
For future handoffs, test imports and one complete optimizer step before
writing integration instructions, and package only dependency-independent code.
