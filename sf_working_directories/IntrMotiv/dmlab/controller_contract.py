"""Fail-closed classification of full-parent controller experiment deltas."""

import json

# Enumerate real parser options. A broad controller_* exemption would hide typos
# and silently authorize new scientific factors.
CONTROLLER_OPTIONS = frozenset(
    {
        "controller_replay_state",
        "controller_learning",
        "controller_her",
        "controller_epsilon",
        "controller_target_updates",
        "controller_replay_capacity",
        "controller_td_positions",
        "controller_decisions_per_update",
        "controller_learning_starts",
        "controller_her_positions",
        "controller_her_loss_coeff",
        "controller_epsilon_decay_decisions",
        "controller_preflight",
        "controller_cache_visual",
    }
)


def preservation_delta(parent, candidate, *, approved_factors=None):
    """Return every config difference and reject unclassified changes.

    Approval descriptions must identify why execution/initialization/seed fields
    differ. Architecture changes are never exempted by a prefix or allowlist.
    Canonical StudySpec remains responsible for run expansion and submission.
    """
    approved_factors = approved_factors or {}
    result = []
    for key in sorted(parent.keys() | candidate.keys()):
        before = parent.get(key)
        after = candidate.get(key)
        # JSON normalization handles tuples restored from serialized configs.
        if (
            key in parent
            and key in candidate
            and json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)
        ):
            continue
        if key in CONTROLLER_OPTIONS:
            classification = "controller_learning"
            reason = "Opt-in controller implementation option"
        elif key in approved_factors and approved_factors[key].strip():
            classification = "explicit_experimental_factor"
            reason = approved_factors[key]
        else:
            raise ValueError(f"Unclassified parent change: {key}: {before!r} -> {after!r}")
        result.append(
            {
                "field": key,
                "parent": before,
                "candidate": after,
                "classification": classification,
                "reason": reason,
                "parent_present": key in parent,
                "candidate_present": key in candidate,
            }
        )
    return result
