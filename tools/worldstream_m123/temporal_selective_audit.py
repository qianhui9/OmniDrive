"""Read-only evidence verification and optional delivery manifest generation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from .benchmark.common import sha256,write
from .selective_answer_reuse_report import lines


def freeze_quality(root):
    files=['tools/worldstream_m123/temporal_selective_report.py','tools/worldstream_m123/selective_answer_reuse_report.py',
           'tools/worldstream_m123/answer_validity_quality.py']
    target=root/'QUALITY_SCOPE_FREEZE.json'
    if target.exists():
        assert all(sha256(p)==h for p,h in json.loads(target.read_text())['files'].items())
        print('QUALITY_ALREADY_FROZEN');return
    assert not (root/'run').exists(),'Cannot freeze quality after TEST begins'
    for p in files:
        dest=root/'frozen_quality_sources'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write(target,dict(files={p:sha256(p) for p in files},before_test=True,
        quality='Independent current visible required facts; UNKNOWN retained, severe errors separate; same metric as previous run',
        new_test='3 disjoint logs,4 contiguous frames each',no_threshold_or_seed_retuning=True))
    print('QUALITY_FROZEN')


def interpret(root):
    """Render the already frozen metrics; no policy or grading changes."""
    get=lambda name:json.loads((root/name).read_text())
    perf=get('abc_comparison.json');q=get('task_quality.json');ab=get('temporal_state_contribution.json')
    result=get('go_no_go.json');fast=get('fast_answer_analysis.json');miss=get('miss_pairing_diagnostic.json')
    temporal=get('temporal_competitor_analysis.json');old=get('old_binding_regression.json')
    def group(condition,execution='all'):
        return next(g for g in q['groups'] if (g['workload'],g['question_count'],g['condition'],g['execution'])==('addressable',3,condition,execution))
    def speed(w,n=3):return next(s for s in perf['summaries'] if (s['workload'],s['question_count'],s['comparison'])==(w,n,'B->C'))
    b,c,h=group('B'),group('C'),group('C','fast_hit')
    cur=next(g for g in ab['quality']['groups'] if (g['condition'],g['question_count'],g['execution'])==('C-current',3,'fast_hit'))
    ap=next(s for s in ab['performance']['summaries'] if s['question_count']==3)
    cost=next(g for g in perf['module_breakdown'] if (g['workload'],g['question_count'],g['condition'])==('addressable',3,'C'))
    policy=get('selected_policy.json');all_policies=get('quality_latency_pareto.json')['policies']
    old_unresolved=any(d['mode']=='new_temporal' and d['proposal']['admitted'] and
        d['proposal']['facts'][0]['semantic_class']=='truck' for d in old['decisions'])
    texts=[
        '1. 已因果复现旧 Right-Rear→Right-Front 差异：当前 Embeddings 和 Draft 完全相同，双向只交换真实 RNG 状态，答案就双向交换。根因是前两个问题被跳过后，共享帧级采样流偏移。',
        '2. 已用 SHA256(Scene,Frame,QuestionIndex,Repeat,ExperimentSeed) 派生请求种子，并用 fork_rng 保存/恢复 CPU 与当前 CUDA RNG；未改 Greedy、温度、Top-p 或长度。隔离后该诊断 B/C 一致，但两者都答错为 left-front；没有通过挑 Seed 消除模型错误。',
        f"3. 有限诊断的3个正常未命中配对全部一致。正式 TEST 的{miss['normal_miss_requests']}个 B/C 正常未命中均使用相同种子，其中{miss['identical_tokens']}个 Token 序列一致。唯一不同处 B 选用了自身历史精确 Draft，C 因前序快速回答没有该目标生成历史，进入普通采样；起始种子相同，后续拒绝采样路径不同。该剩余差异没有额外因果重放，不能把它说成已隔离的纯 FP16 效应，当前事实标签也是 UNKNOWN。详见 miss_pairing_diagnostic.json。",
        '4. 原来的 Car/Truck 错误是“当前已检测对象中最近”不等于真实最近；近处 Car 未成为当前合法观测候选，远处 Truck 的0.731检测分数不能证明目标选择正确。',
        f"5. 新机制对已观测竞争、边界和历史潜在遗漏做风险判断；DEV及本轮 TEST 已评分快速回答没有错误。但旧 Car/Truck 案例{'仍未解决' if old_unresolved else '未在该诊断复现'}。它是该流首帧，没有合法历史；当前绑定和时序机制都仍会输出Truck。不能根据零个新 TEST 命中错误就声称修复了所有漏检。",
        f"6. 正式数据识别{temporal['unique_frame_questions']}个独立 Frame/Question 潜在竞争情形，展开重复及1Q/3Q共{len(temporal['events'])}条事件。历史位置通过真实 Ego 变换和显式不确定运动包络得到，只形成风险，不作为当前事实。",
        f"7. 尚未证明独立时序收益。三问 Current-Only 命中{cur['requests']}/108，Temporal-Aware命中{h['requests']}/108；已评分质量均为{c['score']:.2%}。少掉的9次命中都没有可靠当前标签。C-current→C-temporal实测比值{ap['e2e_speedup']:.4f}，时延差CI为{ap['delta_scene_CI95_ms']}ms，跨零。当前检测逐项一致，故可确认时序项改变了决策，但无法确认改善了质量或效率。",
        f"8. 新三问命中覆盖{h['requests']}/{h['requests']+group('C','miss')['requests']}={h['requests']/c['requests']:.2%}；上一轮为48/81=59.26%。测试流不同，只能描述下降，不能作为同数据因果对比。",
        f"9. 正式A/B/C真实跳过{fast['skips']}次M3，全部有独立Forward钩子验证零语言Forward、零输入展开及Prefill/Decode不执行。分阶段的DEV和消融跳过数见integrity_audit.json，不混入正式45次。",
        f"10. 三问C平均每请求：解析{cost['question_fact_extraction_wall_ms_per_question']:.6f}ms，当前绑定{cost['current_binding_wall_ms_per_question']:.6f}ms，时序风险{cost['temporal_risk_wall_ms_per_question']:.6f}ms。时序计算纯CPU，额外GPU/LLM Forward为0。当前绑定与时序成本已包含在有效性检查中，不能重复相加。",
        f"11. 新三问B→C为{speed('addressable')['e2e_speedup']:.6f}×，低于旧不同测试集的1.590×；仍有三个场景一致正收益，不能宣称保留了旧加速幅度。",
        f"12. 自然3Q {speed('natural')['e2e_speedup']:.6f}×，固定3Q {speed('fixed')['e2e_speedup']:.6f}×，均无新增收益；自然1Q {speed('natural',1)['e2e_speedup']:.6f}×，固定1Q {speed('fixed',1)['e2e_speedup']:.6f}×，有约2%的实测退化。决策CPU成本明显小于全部E2E差，不能把整段退化都归因于解析或时序分析。",
        f"13. 三问B→C完整M123加速{speed('addressable')['e2e_speedup']:.6f}×，场景配对E2E差CI {speed('addressable')['delta_scene_CI95_ms']}ms；单问{speed('addressable',1)['e2e_speedup']:.6f}×的CI跨零。",
        f"14. 新三问已评分事实质量B={b['score']:.2%}，C={c['score']:.2%}，提高{(c['score']-b['score'])*100:.2f}pp；评分覆盖{c['scored']}/{c['requests']}。快速回答{h['scored']}/{h['requests']}可评分且全部正确，其余UNKNOWN。旧C=82.35%来自另一测试集，不能据此判断方法因果变化。",
        f"15. 相对B新增可证实严重错误为{result['new_severe_errors']}。C仍有{c['severe']}条严重输出（含重复），均在正常目标模型生成路径；没有已评分快速回答严重错误。旧诊断B/C同错也保留，不能把“没有新增”写成“系统没有错误”。",
        f"16. DEV的1/2/3pp预算均选同一实际配置：支持阈值{policy['min_confidence']}、绑定风险上限{policy['max_binding_risk']}、净收益阈值{policy['min_net_gain_ms']}ms。只有一个选中工作点，不是三个独立Pareto策略；DEV线上均值{policy['online_dev_metrics']['mean_e2e_ms']:.3f}ms/帧，已评分相对损失{policy['online_dev_metrics']['loss']:.6f}。TEST未调阈值。",
        '17. 本轮证明的是受限问题域的当前帧事实驱动语言计算省略，以及跨问题RNG解耦。跨帧状态确实参与决策，但独立质量或时延收益尚未证明；历史风险过滤的额外样本恰好缺少可靠标签。',
        f"18. 最终{result['status']}，适用范围仅限已评分M1-Addressable事实问答。可以作为受限M3执行模块的系统证据，但不足以支撑“时序世界状态已有独立优势”或广泛自然驾驶QA/安全主张。默认研究开关继续关闭，不自动部署。"]
    (root/'result_answers.md').write_text('# WRSAR v2：18项最终回答\n\n'+'\n\n'.join(texts)+'\n')
    (root/'delivery_findings.md').write_text('# Measured interpretation\n\n'+'\n\n'.join(texts[i-1] for i in (1,3,5,7,12,15,17,18))+'\n')
    path=root/'implementation_summary.md';text=path.read_text();marker='\n## Delivery interpretation\n'
    if marker in text:text=text.split(marker)[0]
    path.write_text(text+marker+'\n'+texts[4]+'\n\n'+texts[6]+'\n\n'+texts[14]+'\n')
    print('INTERPRETATION_RENDERED_FROM_FROZEN_METRICS')


def verify(root,checkpoint=False,manifest=False):
    policy=json.loads((root/'POLICY_FREEZE.json').read_text())
    assert sha256(root/'selected_policy.json')==policy['selected_policy_sha256']
    assert sha256('tools/worldstream_m123/selective_answer_reuse_report.py')==policy['grader_sha256']
    scope=json.loads((root/'QUALITY_SCOPE_FREEZE.json').read_text())
    assert all(sha256(p)==h for p,h in scope['files'].items())
    protocol=json.loads((root/'protocol_freeze.json').read_text())
    assert sha256(root/'experiment_plan.json')==protocol['plan_sha256']
    plan=json.loads((root/'experiment_plan.json').read_text())
    assert len({s['log_id'] for s in plan['test_scenes']})==3
    assert not set(plan['prior_excluded_log_ids'])&{s['log_id'] for s in plan['test_scenes']}
    assert not {s['log_id'] for s in plan['development_scenes']} & {s['log_id'] for s in plan['test_scenes']}
    counts={};skips=Counter();checks={};checkpoint_identity=set()
    for stage in ('diagnostic','shadow','pareto','run','ablation'):
        status=json.loads((root/stage/'run_status.json').read_text());assert status['status']=='COMPLETED'
        checkpoint_identity.add(status['checkpoint_sha256'])
        sources=json.loads((root/stage/'execution_sources.json').read_text())['sources']
        for p,h in sources.items():
            p=Path(p);relative=p.relative_to(Path.cwd()) if p.is_absolute() else p
            assert sha256(root/stage/'executed_sources'/relative)==h
            # Inference sources remained fixed throughout every final stage.
            if stage!='diagnostic':assert sha256(p)==h
        checks[stage]=dict(executed_sources_verified=len(sources))
        rows=lines(root/stage/'requests.jsonl');counts[stage]=len(rows)
        assert counts[stage]==status['full_model_requests']
        for r in rows:
            assert r['m1_commit_completed'] and r['m2_completed'] and r['execution_completed']
            assert r['observed_llm_forward_calls']==sum(q['llm_forward_calls'] for q in r['requests'])
            if r['condition']!='A':
                for q in r['requests']:
                    if q['llm_skipped']:
                        p=q['fast_answer'];skips[stage]+=1
                        assert q['llm_forward_calls']==0 and q['layout_preparation_ms']==0
                        assert not q['prefill_completed'] and not q['decode_completed']
                        assert p['supported'] and p['admitted'] and p['facts']
                        for f in p['facts']:
                            assert f['source_frame']==r['frame_id'] and f['source_timestamp']==r['timestamp']
                            assert f['evidence_origin']=='actual_perception'
                    else:
                        assert q['executor']=='PrefillExecutor' and q['prefill_completed'] and q['decode_completed']
                        if stage!='diagnostic':assert q['question_rng']['enabled']
            if stage=='shadow':assert all(not q['llm_skipped'] for q in r['requests'])
    assert len(checkpoint_identity)==1
    cp=dict(expected_sha256=next(iter(checkpoint_identity)),actual_hash_verified=False)
    if checkpoint:
        actual=sha256('ckpts/iter_10548.pth');assert actual==cp['expected_sha256']
        cp.update(actual_sha256=actual,actual_hash_verified=True)
        write(root/'checkpoint_verification.json',cp)
    elif (root/'checkpoint_verification.json').exists():
        cp=json.loads((root/'checkpoint_verification.json').read_text());assert cp['actual_hash_verified']
    ablation=json.loads((root/'temporal_state_contribution.json').read_text())
    assert ablation['all_current_worlds_equal'],'Temporal ablation current M1 evidence differs'
    rng=json.loads((root/'rng_pairing_diagnostic.json').read_text())
    assert rng['historical_error_exactly_reproduced'] and rng['same_input_draft_rng_crossover_changes_legacy_output']
    assert rng['isolated_rng_crossover_same_output']
    assert all(e['same_input_embeddings'] and e['same_draft_candidates'] for e in rng['pairs'])
    assert all(e['same_rng_before'] and e['same_output'] for e in rng['pairs'] if e['mode']=='isolated')
    misses=json.loads((root/'miss_pairing_diagnostic.json').read_text());assert misses['all_same_seed']
    validation=json.loads((root/'validation_gpu/summary.json').read_text());assert validation['exit_code']==0 and validation['CUDA_available']
    result=dict(status='PASS',sources=checks,full_M123_calls=sum(counts.values()),stages=counts,
        extra_M3_only_diagnostic_generations=8,actual_skips=dict(skips),checkpoint=cp,
        frozen_policy_and_quality=True,new_test_scene_logs_verified=True,
        temporal_ablation_current_detections_equal=True,production_miss_seeds_equal=True,
        actual_language_forward_hook_matches=True,all_fast_facts_current_perception=True,
        regression_tests=validation['counts'],deployment_enabled=False)
    if manifest:
        write(root/'integrity_audit.json',result)
        files={}
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.name=='artifact_manifest.json' or any('q_r' in part for part in p.parts):continue
            if p.suffix in ('.pt','.pth','.pkl','.npy','.npz'):raise AssertionError('Duplicated tensor in delivery: '+str(p))
            files[str(p)]=dict(bytes=p.stat().st_size,sha256=sha256(p))
        write(root/'artifact_manifest.json',dict(status='VERIFIED_DELIVERY',files=files,total_bytes=sum(v['bytes'] for v in files.values())))
    if (root/'artifact_manifest.json').exists():
        m=json.loads((root/'artifact_manifest.json').read_text())
        assert all(Path(p).stat().st_size==v['bytes'] and sha256(p)==v['sha256'] for p,v in m['files'].items())
        result['manifest_files_verified']=len(m['files'])
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/m3_temporal_selective_bypass_run01'))
    parser.add_argument('--checkpoint',action='store_true')
    parser.add_argument('--manifest',action='store_true')
    parser.add_argument('--freeze-quality',action='store_true')
    parser.add_argument('--interpret',action='store_true')
    args=parser.parse_args()
    if args.freeze_quality:freeze_quality(args.root)
    elif args.interpret:interpret(args.root)
    else:verify(args.root,args.checkpoint,args.manifest)
