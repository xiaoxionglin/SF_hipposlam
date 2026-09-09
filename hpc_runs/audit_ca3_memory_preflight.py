"""Scientific runtime gates for the finite-memory study; no training execution."""
import argparse
import json
import math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.discovery import discover_run_directories


def audit(study, root, required_frames):
    directories = discover_run_directories(study, root)
    records = []
    for run in study.expand_runs():
        values = {}
        for directory in sorted({p.parent for p in directories[run.name].rglob('events.out.tfevents.*')}):
            accumulator = EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()
            for tag in accumulator.Tags()['scalars']:
                values.setdefault(tag, []).extend(accumulator.Scalars(tag))
        tags = {
            'frames': 'train/env_steps',
            'gate_violation': 'intrmotiv/memory/gate_violation_fraction',
            'replay_mismatch': 'intrmotiv/memory/replay_mismatch',
            'goal_active': 'intrmotiv/memory/goal_active_fraction',
            'goal_hit': 'intrmotiv/memory/goal_hit_fraction',
            'novel_onset': 'intrmotiv/memory/novel_onset_fraction',
            'onset': 'intrmotiv/memory/onset_fraction',
            'ppo_gradient': 'intrmotiv/dg/gradient/ppo_norm',
            'encoder_gradient': 'intrmotiv/dg/gradient/encoder_norm',
            'reward_nonzero': 'intrmotiv/reward/intrinsic_nonzero_fraction',
        }
        record = {'run': run.name, 'errors': []}
        for key, tag in tags.items():
            if key == 'replay_mismatch' and run.context['cell_control'] != 'goal':
                record[key] = {'last': None, 'max': 0, 'samples': 0, 'applicable': False}
                continue
            events = sorted(values.get(tag, []), key=lambda event: event.wall_time)
            record[key] = {'last': events[-1].value, 'max': max(e.value for e in events),
                           'samples': len(events)} if events else None
            if not events or any(not math.isfinite(e.value) for e in events):
                record['errors'].append(f'{key}: missing or nonfinite')
        if record['errors']:
            records.append(record)
            continue
        if record['frames']['max'] < required_frames:
            record['errors'].append('incomplete training')
        for key in ('gate_violation', 'replay_mismatch'):
            if record[key]['max'] > 1e-7:
                record['errors'].append(f'{key}: nonzero')
        for key in ('onset', 'encoder_gradient'):
            if record[key]['max'] <= 0:
                record['errors'].append(f'{key}: never positive')
        if run.context['cell_gradient'] == 'stop' and record['ppo_gradient']['max'] > 1e-7:
            record['errors'].append('STOP has PPO-to-DG gradient')
        if run.context['cell_gradient'] == 'joint' and record['ppo_gradient']['max'] <= 0:
            record['errors'].append('JOINT has no PPO-to-DG gradient')
        if run.context['cell_control'] == 'goal':
            for key in ('goal_active', 'goal_hit'):
                if record[key]['max'] <= 0:
                    record['errors'].append(f'{key}: never positive')
        elif record['reward_nonzero']['max'] <= 0:
            record['errors'].append('flat decoder reward is always zero')
        records.append(record)
    return {'schema': study.raw['schema'], 'workflow_version': study.declared_workflow_version,
            'study_sha256': study.fingerprint, 'required_frames': required_frames,
            'passed': all(not r['errors'] for r in records), 'runs': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('root', type=Path)
    parser.add_argument('--required-frames', type=int, default=2000000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_study(args.study), args.root, args.required_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
