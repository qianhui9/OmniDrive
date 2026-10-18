"""Validate staged artifacts and retain an immutable delivery source manifest."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .audit import write

ROOT=Path('outputs/system_optimization_v4')
REQUIRED=('task_dependency_audit.md','task_state_feasibility.json','theoretical_benefit_bound.json',
    'source_geometry_error_analysis.json','task_fact_results.json','task_fact_risk_coverage.json',
    'task_state_consumption.json','task_quality_results.json','effective_token_reduction.json',
    'llm_computation_savings.json','auto_cost_results.json','streaming_replay_results.json',
    'system_ablation.json','go_nogo_decisions.json','experiment_report.md','commands.sh')


def main():
    missing=[name for name in REQUIRED if not (ROOT/name).exists()]
    if missing:raise AssertionError('Missing required artifacts: '+str(missing))
    for path in ROOT.glob('*.json'):
        json.loads(path.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    states={str(p.parent):json.loads(p.read_text())['status'] for p in ROOT.glob('*/status.json')}
    assert len(states)==18 and all(s=='completed' for s in states.values())
    assert json.loads((ROOT/'gpu_status.json').read_text())['status']=='completed'
    assert json.loads((ROOT/'model_manifest.json').read_text())['weights_unchanged']
    old=json.loads((ROOT/'previous_result_hashes.json').read_text())
    unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old.items()}
    assert all(unchanged.values())
    log=(ROOT/'logs/final_gpu_tests.log').read_text()
    tests=re.search(r'(\d+) passed(?:, (\d+) skipped)?',log)
    assert tests and ' failed' not in log and 'ERROR ' not in log
    monitors=[json.loads(line) for line in (ROOT/'logs/environment_monitor.jsonl').read_text().splitlines()]
    violations=[];rss=[]
    for sample in monitors:
        for thread in sample['threads']:
            cpus=set()
            for part in thread['affinity'].split(','):
                bounds=list(map(int,part.split('-')))
                cpus.update(range(bounds[0],bounds[-1]+1))
            if cpus & {4,5}:violations.append(thread)
        current=sample.get('process',{}).get('VmRSS')
        if current:rss.append(int(current.split()[0])*1024)
    assert not violations
    write(ROOT/'resource_summary.json',dict(monitor_samples=len(monitors),thread_affinity_violation_count=0,
        process_rss_min_bytes=min(rss) if rss else None,process_rss_max_bytes=max(rss) if rss else None,
        actual_stream_queue_memory_growth_bytes=None,
        scope='RSS belongs to one shared model process across all methods, including post-timer audits; not a per-method net memory or queue-growth estimate',
        source='logs/environment_monitor.jsonl'))
    source={}
    paths=list(Path('worldstream').rglob('*.py'))+list(Path('tools/system_optimization').glob('*.py'))
    paths+=list(Path('tests').rglob('test_*facts.py'))+[Path('tests/worldstream_m3/test_v4_admission.py'),Path('tests/worldstream_m3/test_streaming_replay.py')]
    for path in paths:
        dest=ROOT/'delivery_source'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        source[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    diff=subprocess.check_output(['git','diff','--','worldstream'],text=True)
    (ROOT/'implementation.patch').write_text(diff)
    write(ROOT/'delivery_manifest.json',dict(status='validated',tests_passed=int(tests.group(1)),
        tests_skipped=int(tests.group(2) or 0),old_v3_results_unchanged=unchanged,
        required_artifact_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in REQUIRED},
        source_sha256=source,run_states=states,
        affinity_monitor_samples=len(monitors),forbidden_cpu_violation_count=0,
        source_scope='Per-run source_manifest.json identifies source at capture time; delivery_source includes later offline reporting and tests',
        inherited_runner_metadata_erratum='Per-run protocol independence text mentions indices 96+ from an inherited template. Actual frozen v4 scene/sample tokens and scene_start values define this experiment; no new claim uses that template sentence.'))
    with (ROOT/'experiment_report.md').open('a') as f:
        f.write(f'\n最终 GPU 可见环境回归：{tests.group(1)} 项通过，{tests.group(2) or 0} 项跳过。18 个场景执行正常完成；权重全文件哈希和 10 个 v3 原始报告/协议哈希保持不变。详见 delivery_manifest.json。\n')
    # The report changed only by the actual final test footer; pin final bytes.
    manifest=json.loads((ROOT/'delivery_manifest.json').read_text())
    manifest['required_artifact_sha256']['experiment_report.md']=hashlib.sha256((ROOT/'experiment_report.md').read_bytes()).hexdigest()
    write(ROOT/'delivery_manifest.json',manifest)
    print(json.dumps(dict(tests_passed=int(tests.group(1)),previous_results_unchanged=True,run_count=len(states))))


if __name__=='__main__':main()
