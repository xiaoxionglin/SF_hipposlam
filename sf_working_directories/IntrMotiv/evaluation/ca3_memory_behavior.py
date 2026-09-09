"""Segment-aware event repetition and physical trajectory diagnostics."""
import numpy as np


def behavior_diagnostics(pose, dg, refractory=8, window=64):
    motifs = {period: [] for period in (1, 2, 3, 4)}
    intervals, efficiency, movement = [], [], []
    events_total = 0
    for _, group in pose.reset_index(drop=True).groupby(['agent', 'num_traj'], sort=False):
        activity = np.asarray(dg)[group.index]
        positions = group[['x', 'y']].to_numpy()
        last_active = np.full(activity.shape[1], -refractory - 1)
        sequence, times = [], []
        for t, values in enumerate(activity):
            positive = values > 0
            candidates = positive & ((t - last_active) > refractory)
            if candidates.any():
                sequence.append(int(np.where(candidates, values, -np.inf).argmax()))
                times.append(t)
            last_active[positive] = t
        events_total += len(sequence)
        intervals.extend(np.diff(times).tolist())
        for period in motifs:
            motifs[period].extend(sequence[t-period:t] == sequence[t:t+period]
                                  for t in range(period, len(sequence)-period+1))
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        movement.extend(steps.tolist())
        for t in range(window, len(positions)):
            path = steps[t-window:t].sum()
            if path > 0:
                efficiency.append(float(np.linalg.norm(positions[t]-positions[t-window]) / path))
    def quantiles(values):
        return np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None
    return dict(dominant_onsets=events_total, onset_interval_decisions_p10_p50_p90=quantiles(intervals),
        physical_step_distance_p10_p50_p90=quantiles(movement),
        window_straightness_p10_p50_p90=quantiles(efficiency), straightness_window_decisions=window,
        stationary_step_fraction=float(np.mean(np.asarray(movement) == 0)) if movement else None,
        **{f'repeated_motif_period_{p}_fraction': float(np.mean(v)) if v else None for p, v in motifs.items()})
