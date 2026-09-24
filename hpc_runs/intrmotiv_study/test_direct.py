from pathlib import Path
import pytest
from hpc_runs.intrmotiv_study.direct import log_status, source_digest, atomic_json


def test_source_digest_detects_source_not_outputs(tmp_path):
    source = tmp_path / 'sample_factory'; source.mkdir()
    (source/'train.py').write_text('a=1')
    first = source_digest(tmp_path)
    (source/'events.log').write_text('changing log')
    assert first == source_digest(tmp_path)
    (source/'train.py').write_text('a=2')
    assert first != source_digest(tmp_path)


def test_log_failure_even_with_progress(tmp_path):
    path = tmp_path/'train.log'
    path.write_text('Total num frames: 65536\nhttps://wandb.ai/entity/project/runs/abc123\nTraceback (most recent call last)')
    result = log_status(path)
    assert result['frames'] == 65536 and result['traceback']
    assert len(result['wandb_urls']) == 1


def test_atomic_json_replaces(tmp_path):
    path=tmp_path/'state.json'
    atomic_json(path, {'state':'pending'})
    atomic_json(path, {'state':'done'})
    assert 'done' in path.read_text()
    assert not list(tmp_path.glob('*.tmp'))


def test_queue_records_failure_and_continues_with_later_runs(tmp_path, monkeypatch):
    import json, sys
    from hpc_runs.intrmotiv_study.direct import run_queue
    monkeypatch.setenv('WANDB_API_KEY', 'test-placeholder-never-sent')
    def probe():
        return {'available_ram_gib':200,'gpus':[{'index':0,'free_mib':64000,'utilization':0}]}
    common=dict(source_root=str(tmp_path),source_sha256=source_digest(tmp_path),gpu_slots=[0],
                schema='test',workflow_version='test',study_sha256='test',manifest_sha256='test')
    success=dict(name='success',target_frames=10,command=[sys.executable,'-c',
                 "print('Total num frames: 10\\nhttps://wandb.ai/e/p/runs/unit_test')"])
    passed=run_queue(dict(common,output_root=str(tmp_path/'pass'),runs=[success]),probe,poll_seconds=.01)
    assert passed[0]['status']=='completed'
    assert passed[0]['wandb_urls']==['https://wandb.ai/e/p/runs/unit_test']
    failure=dict(name='failure',target_frames=10,command=[sys.executable,'-c',"raise RuntimeError('test failure')"])
    with pytest.raises(RuntimeError,match='run failed'):
        run_queue(dict(common,output_root=str(tmp_path/'fail'),runs=[failure,success]),probe,poll_seconds=.01)
    states=json.loads((tmp_path/'fail/direct_execution/state.json').read_text())
    assert [s['status'] for s in states]==['failed','completed']
