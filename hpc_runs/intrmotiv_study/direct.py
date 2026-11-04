"""Auditable direct Sample Factory queue for GPU workstations (no scheduler).

Print a manifest first, then pass its SHA to --execute. Scientific arguments
come exclusively from StudySpec. Slots specify physical GPUs, including repeats.
Only processes created by this queue are signalled after a failed child run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .spec import load_study


def source_digest(root):
    """Fingerprint executable Python and environment Lua, excluding outputs."""
    digest = hashlib.sha256()
    for directory in ('sample_factory', 'sf_working_directories/IntrMotiv', 'hpc_runs/intrmotiv_study'):
        for path in sorted((Path(root) / directory).rglob('*')):
            if path.suffix not in ('.py', '.lua') or '__pycache__' in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def make_manifest(study, source, slots):
    root = Path(study.workspace_root).resolve()
    output = Path(study.output_root).resolve()
    if not output.is_relative_to(root) or not Path(source).resolve().is_relative_to(root):
        raise ValueError('Source and outputs must be in the declared workspace')
    runs = []
    for run in study.expand_runs():
        args = list(run.args)
        if any(a.startswith(('--experiment=', '--train_dir=')) for a in args):
            raise ValueError('Direct queue owns experiment and train_dir arguments')
        command = [sys.executable, '-m', 'sf_working_directories.IntrMotiv.dmlab.train_hipposlam',
                   *args, f'--experiment={run.name}', f'--train_dir={output}']
        tokens = dict(a[2:].split('=', 1) for a in args if a.startswith('--') and '=' in a)
        if tokens.get('with_wandb', '').lower() != 'true':
            raise ValueError('This queue requires online W&B')
        for key in ('wandb_dir', 'dmlab_level_cache_path', 'online_spatial_output_root'):
            if key in tokens and not Path(tokens[key]).resolve().is_relative_to(root):
                raise ValueError(f'{key} escapes workspace')
        runs.append({**run.as_dict(), 'command': command, 'target_frames': int(tokens['train_for_env_steps'])})
    result = {**study.provenance(), 'source_root': str(source), 'source_sha256': source_digest(source),
              'output_root': str(output), 'gpu_slots': slots, 'runs': runs}
    result['manifest_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def log_status(path):
    text = path.read_text(errors='replace') if path.exists() else ''
    frames = re.findall(r'Total num frames: (\d+)', text)
    return {'frames': int(frames[-1]) if frames else 0,
            'traceback': 'Traceback (most recent call last)' in text,
            'wandb_urls': sorted(set(re.findall(r'https://wandb.ai/[^\s\x1b]+/runs/[a-zA-Z0-9_-]+', text)))}


def run_queue(manifest, resource_probe, *, poll_seconds=10):
    import psutil
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Supply W&B credentials in the environment')
    output = Path(manifest['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    audit = output / 'direct_execution'
    audit.mkdir()  # Atomic duplicate-launch exclusion; explicit recovery after a crash.
    atomic_json(audit / 'manifest.json', manifest)
    states = [dict(name=r['name'], status='pending') for r in manifest['runs']]
    active = {}
    failed = False
    with (audit / 'resources.jsonl').open('w') as resource_log:
        while any(s['status'] in ('pending', 'running') for s in states):
            resource = resource_probe()
            resource_log.write(json.dumps(resource) + '\n')
            resource_log.flush()
            for index, owned in list(active.items()):
                state, spec = states[index], manifest['runs'][index]
                state.update(log_status(Path(state['log'])))
                code = owned['process'].poll()
                state['returncode'] = code
                if state['traceback'] and code is None and not owned.get('stop_sent'):
                    owned['process'].send_signal(signal.SIGINT)
                    owned['stop_sent'] = time.time()
                    failed = True
                if code is not None:
                    owned['handle'].close()
                    ok = code == 0 and not state['traceback'] and state['frames'] >= spec['target_frames'] and bool(state['wandb_urls'])
                    state.update(status='completed' if ok else 'failed', finished_at=time.time())
                    failed |= not ok
                    del active[index]
            free_slots = [i for i in range(len(manifest['gpu_slots'])) if i not in {o['slot'] for o in active.values()}]
            if free_slots:
                # Leave at least 64 GiB host and 16 GiB GPU headroom. The G500
                # DG sweep measured about 55 GiB of incremental host memory for
                # the fourth 32x16 run near full load, so reserve that amount
                # for every admission instead of the old optimistic 8 GiB.
                host_run_reserve_gib = 55
                ram = resource['available_ram_gib']
                gpu_free = {int(g['index']): g['free_mib'] for g in resource['gpus']}
                gpu_util = {int(g['index']): g['utilization'] for g in resource['gpus']}
                for slot in free_slots:
                    index = next((i for i, s in enumerate(states) if s['status'] == 'pending'), None)
                    if index is None:
                        break
                    gpu = manifest['gpu_slots'][slot]
                    if ram < 64 + host_run_reserve_gib or gpu_free.get(gpu, 0) < 32768 or gpu_util.get(gpu, 100) > 80:
                        continue
                    if source_digest(manifest['source_root']) != manifest['source_sha256']:
                        raise RuntimeError('Source changed after manifest review; refusing launch')
                    spec, state = manifest['runs'][index], states[index]
                    if (output / spec['name']).exists():
                        raise FileExistsError(f"Refusing implicit resume: {spec['name']}")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE='online',
                               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               PYTHONPATH=manifest['source_root'], INTRMOTIV_SOURCE=manifest['source_root'])
                    for key in ('schema', 'workflow_version', 'study_sha256', 'source_sha256', 'manifest_sha256'):
                        env['INTRMOTIV_' + key.upper()] = str(manifest[key])
                    log = audit / (spec['name'] + '.log')
                    handle = log.open('x')
                    process = subprocess.Popen(spec['command'], cwd=manifest['source_root'], env=env,
                                               stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    state.update(status='running', pid=process.pid, create_time=psutil.Process(process.pid).create_time(),
                                 gpu=gpu, log=str(log), started_at=time.time())
                    active[index] = dict(process=process, handle=handle, slot=slot)
                    ram -= host_run_reserve_gib
                    gpu_free[gpu] -= 16384
            atomic_json(audit / 'state.json', states)
            if active or any(s['status'] == 'pending' for s in states):
                time.sleep(poll_seconds)
    if failed:
        raise RuntimeError('A run failed; inspect direct_execution/state.json')
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--execute', metavar='REVIEWED_MANIFEST_SHA')
    args = parser.parse_args()
    manifest = make_manifest(load_study(args.study), args.source.resolve(), args.gpus)
    print(json.dumps(manifest, indent=2), flush=True)
    if args.execute:
        if args.execute != manifest['manifest_sha256']:
            parser.error('Manifest differs from print-only review')
        from hpc_runs.hosts.g500.profile_training import resources
        run_queue(manifest, resources)


if __name__ == '__main__':
    main()
