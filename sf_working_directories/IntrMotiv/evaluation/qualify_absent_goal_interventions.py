"""Qualify command interventions against independently collected DG field maps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hpc_runs.intrmotiv_study.spatial_contract import _component_labels as label
from sf_working_directories.IntrMotiv.evaluation.absent_goal_interventions import summarize_commands


def canonical_components(maps, occupancy):
    masks = np.zeros_like(maps, dtype=bool)
    for target in range(maps.shape[-1]):
        rate = np.nan_to_num(maps[..., target])
        peak = rate.max()
        if peak <= 0:
            continue
        labels, count = label((rate >= .5 * peak) & (occupancy > 0))
        if count:
            masses = [rate[labels == component].sum() for component in range(1, count + 1)]
            masks[..., target] = labels == 1 + int(np.argmax(masses))
    return masks


def qualify(rows, masks, bounds=(100., 2000., 100., 2000.)):
    qualified = []
    xmin, xmax, ymin, ymax = bounds
    for row in rows:
        times = [0] * masks.shape[-1]
        for elapsed, target, x, y in row['events']:
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
            ix = min(masks.shape[0] - 1, int((x - xmin) / (xmax - xmin) * masks.shape[0]))
            iy = min(masks.shape[1] - 1, int((y - ymin) / (ymax - ymin) * masks.shape[1]))
            if masks[ix, iy, int(target)] and times[int(target)] == 0:
                times[int(target)] = int(elapsed)
        qualified.append({**row, 'first_onset_times': times})
    return qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trials', type=Path)
    parser.add_argument('fields', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    provenance = json.loads(args.trials.with_name('intervention_summary.json').read_text())
    frame = pd.read_csv(args.trials)
    rows = frame.to_dict('records')
    for row in rows:
        for key in ('events', 'first_onset_times'):
            row[key] = json.loads(row[key])
    with np.load(args.fields, allow_pickle=False) as fields:
        if Path(str(fields['checkpoint'])).resolve() != Path(provenance['checkpoint']).resolve():
            raise ValueError('Field map and intervention must refer to the same checkpoint')
        if not str(fields.get('observation_panel', '')):
            raise ValueError('Spatial qualification requires the independent common-observation panel')
        maps = fields['raw_dg_rate_maps']
        masks = canonical_components(maps, fields['occupancy'])
    summary = summarize_commands(qualify(rows, masks), masks.shape[-1], provenance.get('horizons', [8, 16, 32, 64]))
    summary['spatially_qualified_ctc'] = summary['ctc_auc']
    summary['spatial_qualification_status'] = 'Independent raw-DG largest-mass half-peak component, canonical 8-connectivity'
    summary['field_eligible_identity_count'] = int(masks.any((0, 1)).sum())
    summary['fields_sha256'] = hashlib.sha256(args.fields.read_bytes()).hexdigest()
    summary['trials_sha256'] = hashlib.sha256(args.trials.read_bytes()).hexdigest()
    summary.update({key: provenance[key] for key in ('schema', 'workflow_version', 'study_sha256',
                    'checkpoint', 'condition', 'seed') if key in provenance})
    args.output.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
