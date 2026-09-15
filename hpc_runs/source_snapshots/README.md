# Scoped runtime source snapshots

`topological_manager_device_20260915.tar.gz` contains the local device-resident
manager update, reference fixtures, tests, and benchmark. Applied to the desktop
runtime; G500 transfer and CUDA validation await authorization. The adjacent
text patch excludes the binary NPZ fixture; the archive includes all five files.
See the [implementation record](../../04_implementation/topological_manager_device_20260915.md)
and adjacent file hashes before applying to another checkout.

`graph_planning_optimization_20260911.tar.gz` and its adjacent reviewable patch
contain the behavior-preserving graph-planning acceleration, regression tests,
and reusable microbenchmark. **Staged only: apply before the next new batch,
not silently during the current DG-capacity comparison.** Both local and
isolated NEMO2 focused suites passed 39 tests; the patch passes live-source
print-only application review. See
[implementation and deployment record](../../04_implementation/graph_planning_optimization_20260911.md)
and the adjacent SHA-256 metadata. The original DG-capacity archive is unchanged.

`persistent_intrinsic_control_hotfix_20260908.tar.gz` preserves the complete
Persistent Intrinsic Control runtime source after the legacy-recruitment /
policy-graph invalidation hotfix. It retains the original implementation files
and adds the focused regression, forced-replacement preflight, exact
W_REF_JOINT seed-123 retry adapter, and checkpoint-pinned C15 recovery adapter.
The pre-hotfix
`persistent_intrinsic_control_20260908.tar.gz` remains unchanged.

Hotfix archive SHA-256:
`a55749e0467713b2447f3f5ccc1c402b910aab5cc59fc028c7ec22584dbbe16c`.

`ca3_memory_novelty_goal_20260907.tar.gz` retains the exact seventeen modified
or added IntrMotiv runtime files for the CA3 finite-memory batch. It is a
102-KiB reproducibility artifact, not a second maintained source checkout.
The deployment target is the existing NEMO2 SF_hipposlam checkout; Sample Factory
and DMLab dependencies are not bundled. Studies and thin audit/smoke adapters
are versioned normally in the surrounding `hpc_runs/` directory.

Archive SHA-256:
`83d5e265d92e78bba211bccca18c996a98deaab0183269f4e4c79423f18c3ea9`.

Inspect or extract into a separate temporary directory when reviewing. Do not
blindly unpack over an evolving runtime checkout: full modified files also
contain pre-existing project changes that must be compared and preserved.
See `06_experiments/ca3_memory_novelty_goal_implementation.md` for semantics,
tests, source provenance, and actual submission records.
