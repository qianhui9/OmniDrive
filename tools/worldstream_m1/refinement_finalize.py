"""Check final-source replay equality and archive actual regression/parity proof."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from .io_utils import write_json
from .refinement_experiments import source_manifest
from .refinement_report import ROOT,readlines,sha


def collect(root,name,methods):
    rows=[]
    complete=list((root/'traces/development').glob('*/complete.json'))
    if len(complete)!=20:raise RuntimeError('Final-source validation must contain all 20 scenes')
    for p in complete:rows.extend(r for r in readlines(p.parent/name) if r['method'] in methods)
    return rows


def junit(path):
    t=ET.parse(path);cases=list(t.iter('testcase'))
    result=dict(tests=len(cases),failures=sum(c.find('failure') is not None for c in cases),
        errors=sum(c.find('error') is not None for c in cases),skipped=sum(c.find('skipped') is not None for c in cases),
        xml=str(path),sha256=sha(path))
    result['passed']=result['tests']-result['failures']-result['errors']-result['skipped']
    return result


def main():
    post=ROOT/'post_hardening_replay';names={'surface_consensus','temporal_object'}
    original=collect(ROOT,'pixels.jsonl',names);replayed=collect(post,'pixels.jsonl',names)
    key=lambda r:(r['method'],r['scene'],r['frame'],r['camera'])
    expected={key(r):r for r in original};actual={key(r):r for r in replayed}
    assert len(expected)==len(original)==len(actual)==2880
    fields=('pixels','stable_pixels','effective_pixels','changed_pixels','unknown_pixels','forced_refresh',
        'accepted_points','false_points','scorable_points','critical_points','missed_critical_points','events')
    pixel_diff=[(k,f) for k in expected for f in fields if expected[k].get(f)!=actual[k].get(f)]
    original=collect(ROOT,'identity.jsonl',{'temporal_object'});replayed=collect(post,'identity.jsonl',{'temporal_object'})
    key=lambda r:(r['method'],r['scene'],r['source_frame'],r['frame'],r['anchor_id'],r['prediction'])
    expected={key(r):r for r in original};actual={key(r):r for r in replayed}
    assert len(expected)==len(original)==len(actual)
    fields=('candidate','correct','source_instance','current_instance','certified','rejection_reasons',
        'qualification_reasons','recovered_short_gap','resolved_competition','measurements')
    identity_diff=[(k,f) for k in expected for f in fields if expected[k].get(f)!=actual[k].get(f)]
    assert not pixel_diff and not identity_diff,(pixel_diff[:3],identity_diff[:3])
    frozen=json.loads((post/'development_source_freeze.json').read_text())['files']
    assert frozen==source_manifest(),'Final-source files changed after replay'
    write_json(ROOT/'post_hardening_equality.json',dict(status='passed',scenes=20,frames=240,
        compared_camera_rows=2880,pixel_or_quality_differences=len(pixel_diff),compared_identity_edges=len(expected),
        identity_measurement_or_decision_differences=len(identity_diff),final_runtime_source=frozen,
        changes='Qualified selective tracking metadata, ring image/ROI receipt bounds, CLI/config path isolation. No thresholds or models changed.',
        full_ablation_source='development_source_freeze.json',final_main_method_source='post_hardening_replay/development_source_freeze.json'))
    parity=json.loads((ROOT/'gpu_parity/results.json').read_text());status=json.loads((ROOT/'gpu_parity/status.json').read_text())
    assert status['status']=='passed' and parity['observe_only_equal']
    assert len(parity['run_metadata']['sample_ids'])==8 and all(parity['comparisons'].values())
    checkpoint=(ROOT/'checkpoint.sha256').read_text().split()[0]
    assert checkpoint=='6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d',checkpoint
    write_json(ROOT/'original_model_parity.json',dict(status='passed',actual_cuda=True,frames=8,
        comparisons=parity['comparisons'],observe_only_equal=True,run_metadata=parity['run_metadata'],
        result_sha256=sha(ROOT/'gpu_parity/results.json'),checkpoint_sha256=checkpoint,
        scope='Original backbone, detection/map outputs, both head memory states. Perception-only, M2/M3 disabled; LLM not executed.',
        approval_service='Two initial read-only GPU checks were not executed due to reviewer capacity failure; subsequent approval succeeded. No bypass; no other task stopped.'))
    final=junit(ROOT/'regression_cuda.xml');assert final['failures']==final['errors']==final['skipped']==0
    assert final['passed']==506
    regression=dict(status='passed',python='/home/qh/miniconda3/envs/OmniDrive/bin/python',
        affinity='0-3,6-31',faulty_cpus_excluded=[4,5],final_cuda=final,
        command='pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q',
        final_runtime_source=frozen)
    for phase,source in [('baseline_cpu',Path('/tmp/m1_refinement_baseline.xml'))]:
        if source.exists():
            dest=ROOT/(phase+'.xml');shutil.copy2(source,dest);regression[phase]=junit(dest)
    if (ROOT/'regression_cpu.xml').exists():regression['intermediate_cpu_before_final_hardening']=junit(ROOT/'regression_cpu.xml')
    for source,name in [(Path('/tmp/m1_refinement_selective_metadata_before.log'),'selective_metadata_failure_before_fix.log'),
                        (Path('/tmp/m1_refinement_ring_before.log'),'ring_receipt_failure_before_fix.log')]:
        if source.exists():shutil.copy2(source,ROOT/name)
    write_json(ROOT/'regression_results.json',regression)
    # The source selected for any future independent validation is the final
    # runtime, with the measured earlier full ablation phase retained separately.
    p=ROOT/'frozen_validation_protocol.json';validation=json.loads(p.read_text())
    validation['development_ablation_source_sha256']=json.loads((ROOT/'development_source_freeze.json').read_text())['files']
    validation['source_sha256']=frozen;validation['final_source_main_method_replay']='post_hardening_equality.json'
    write_json(p,validation)
    files=[*Path('worldstream/m1').glob('*.py'),*Path('tools/worldstream_m1').glob('*.py'),
        *Path('tests/worldstream_m1').glob('*.py'),Path('configs/worldstream_m1_quality_coverage_refinement.yaml')]
    write_json(ROOT/'source_manifest.json',dict(baseline_head='78c3a5e',final_files={str(p):sha(p) for p in sorted(files)},
        final_runtime=frozen,numeric_ablation_source='development_source_freeze.json',
        final_source_same_input_revalidation='post_hardening_equality.json',checkpoint_sha256=checkpoint,
        data_manifest_sha256=sha(ROOT/'data_manifest.json'),experiment_protocol_sha256=sha(ROOT/'experiment_protocol.json'),
        commands_sha256=sha(ROOT/'commands.sh')))
    for name,expected_hash in json.loads((ROOT/'diagnostic_freeze.json').read_text()).items():
        assert sha(ROOT/name)==expected_hash,name
    for name,expected_hash in json.loads((ROOT/'historical_artifact_pins.json').read_text()).items():
        assert sha(Path(name))==expected_hash,name
    write_json(ROOT/'artifact_integrity.json',dict(frozen_diagnostics_unchanged=True,pinned_previous_results_unchanged=True,
        final_main_methods_equal=True,final_cuda_tests_passed=True,actual_original_model_parity_passed=True))
    path=ROOT/'experiment_report.md';text=path.read_text().split('\n<!-- FINAL_VALIDATION -->')[0]
    text+='''
<!-- FINAL_VALIDATION -->

## 最终源码与真实工程验证

最终源码完整 CUDA 回归 **506/506 通过，无跳过**。RTX 3090、PyTorch 1.13.1+cu117 上真实运行原模型关闭/开启 M1 各 8 帧；Backbone、Head Memory、感知输出摘要全部一致。Checkpoint SHA256 与原值一致。验证限于 Perception，未执行 LLM。

最终源码的两个主方法又在同一批 20 Scene / 240 Frame 上实际复跑。2880 个相机行及全部 Temporal Object 候选边的测量、决定、独立评分完全相同。源码阶段、配置文件迁移和边界回执修补均保留清单，未覆盖原始消融记录；见 `post_hardening_equality.json`。

旧 Boundary Guard 的实际记录显示：558 个有该字段的相机行中，13 行触发，共排除 76 个候选像素，但旧 on/off 消融最终结果完全相同，说明这些像素未成为最终平面支持。新方法的 guard on/off 也没有改变可评分正确/错误接受，主配置保持关闭。

逐候选参考点的补充审计显示，旧“关键点不足”单元拒绝 1188 个测量正确、323 个测量错误的点。这说明直接放松关键点门槛会同时接纳错误；统计不是对每条规则单独关闭的因果证明，实际配对消融另列。

区域结果仍为 No-Go：有效面积 0.325636% 未超过 Conservative 0.331562%；FAR 28.405797% 违反 5% 门槛；最差关键事件 ROI 刷新只有 61.111111%，且可评分接受仅支持 2 Scene / 2 Log。独立资格仍不足。身份增加 4 条正确链接、2 个唯一实例，但正确支持仍为 6 Scene / 6 Log，不可评分仍为 71；Hybrid 仅 1 个接受 Log 完全可评分。Surface-only 的 4 条正确链接来自 3 个 Log，也不足以认证。

所有测试和实验均用 `taskset -c 0-3,6-31`，避开故障 CPU 4/5；未终止其他 GPU 用户进程。早先 GPU 检查因自动审批容量问题未执行，之后审批成功，最终 GPU 验证已实际完成。
'''
    path.write_text(text)
    print('final validation passed:',final['passed'],'CUDA tests, 8 parity frames, 2880 replay camera rows')


if __name__=='__main__':main()
