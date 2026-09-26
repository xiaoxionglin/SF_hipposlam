"""Frame-based evaluation milestones shared by IntrMotiv learners."""


def checkpoint_targets(cfg):
    """Return explicit targets or eight roughly even points over the planned run.

    An empty setting retains the historical no-frame-milestones behavior for
    saved runs. ``auto`` belongs to new runs and follows their frame budget.
    """
    setting = str(getattr(cfg, "checkpoint_frame_targets", "") or "").strip()
    if setting.lower() == "auto":
        total = int(cfg.train_for_env_steps)
        if total <= 0:
            raise ValueError("automatic checkpoint targets require positive train_for_env_steps")
        return tuple(sorted({(total * index + 7) // 8 for index in range(1, 9)}))
    if not setting:
        return ()
    try:
        targets = tuple(int(part.strip()) for part in setting.split(","))
    except ValueError as error:
        raise ValueError("checkpoint_frame_targets must be 'auto' or comma-separated integers") from error
    if any(target <= 0 for target in targets) or tuple(sorted(set(targets))) != targets:
        raise ValueError("checkpoint_frame_targets must be increasing positive integers")
    return targets
