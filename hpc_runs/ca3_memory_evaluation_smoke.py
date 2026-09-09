"""Bounded Slurm-only integration smoke for panels and graph-free probes."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.discovery import discover_run_directories
from sf_working_directories.IntrMotiv.evaluation.place_fields import rollout_dg, load_policy_env
from sf_working_directories.IntrMotiv.evaluation.absent_goal_interventions import run_absent_goal_interventions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DMLab integration smoke must run inside Slurm')
    study = load_study(args.study)
    directories = discover_run_directories(study, args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    panel = args.output / 'observations.npz'
    runs = study.expand_runs()
    baseline = next(run for run in runs if run.context['cell'] == 'F_D_BASE')
    first = rollout_dg(directories[baseline.name], 128, True, 0, record_panel=panel)
    second = rollout_dg(directories[baseline.name], 128, True, 0, checkpoint_path=first[1], replay_panel=panel)
    np.testing.assert_allclose(first[3], second[3], rtol=0, atol=0)
    np.testing.assert_allclose(first[4], second[4], rtol=0, atol=0)
    np.testing.assert_allclose(first[2][['x','y','z']], second[2][['x','y','z']])
    summaries = {'panel_exact_replay': True, 'panel_samples': len(first[3])}
    for run in runs:
        if run.context['cell_control'] != 'goal':
            continue
        cfg, env, info, actor, checkpoint, device = load_policy_env(directories[run.name], 100000, False, 0)
        trials, summary = run_absent_goal_interventions(cfg, env, info, actor, checkpoint, device,
                                                      10000, starts=2, prefix_length=16, max_commands=2)
        if summary['starts_evaluated'] < 1 or summary['completed_trials'] < 2:
            raise RuntimeError('No valid matched intervention start completed')
        summaries[run.name] = summary
        trials.to_csv(args.output / (run.name + '_trials.csv'), index=False)
    (args.output / 'smoke_summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
