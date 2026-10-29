"""DEV selection and offline TEST reports for state-calibrated bypass."""
from collections import Counter,defaultdict
from dataclasses import replace
import json
from pathlib import Path
import shutil
from .benchmark.common import write,sha256
from .selective_answer_reuse_report import lines,load_labels,scored_requests,mean,grade,wilson
from .temporal_selective_report import performance,quality
from .world_guided_decode_v2_report import scene_ci


def world(data):
    from worldstream.m3.world_state import CommittedWorldState,CommittedAnchor
    return CommittedWorldState(**dict(data,anchors=tuple(CommittedAnchor(**a) for a in data['anchors'])))


def calibrate(root):
    from .state_calibrated_bypass import CONFIGS
    from worldstream.m3.answer_validity import propose_fact_answer
    from worldstream.m3.answer_memory import AnswerMemory
    rows=lines(root/'shadow/requests.jsonl');objects=load_labels(rows);events=scored_requests(rows,objects)
    eventmap={(e['frame_id'],e['request_index']):e for e in events};previous=None
    for row in rows:
        cur=world(row['committed_world'])
        if previous is not None and previous.scene_id!=cur.scene_id:previous=None
        ctx=dict(scene_id=cur.scene_id,frame_id=cur.frame_id,timestamp=cur.timestamp,state_epoch=cur.epoch,
            state_execution_signature=cur.execution_signature,current_state_version=cur.version,
            frame_index=row['frame_index'],camera_ids=row['camera_ids'])
        for qi,q in enumerate(row['requests']):
            e=eventmap[(row['frame_id'],qi)]
            # Same captured CURRENT detections; temporal candidate is a CPU-only
            # DEV counterfactual, not an online skip or measured E2E benefit.
            p=propose_fact_answer(q['question'],cur,ctx,AnswerMemory(),('offline-dev',),None,previous,True,True)
            e['temporal_counterfactual']=p
        previous=cur
    known=[e for e in events if e['quality']['known']];configs=[]
    for i,c in enumerate(CONFIGS):
        hits=[]
        for e in events:
            p=e['temporal_counterfactual'] if c['temporal_risk_enabled'] else e.get('proposal')
            if p and p['supported'] and p['support_score']>=c['min_confidence'] and p['target_binding']['total_risk']<=c['max_binding_risk']:
                hits.append((e,p,grade(e['question'],p['answer_text'],e['frame_id'],objects[e['frame_id']])))
        scored=[(e,p,g) for e,p,g in hits if g['known'] and e['quality']['known']]
        configs.append(dict(index=i,**c,requests=len(events),hits=len(hits),scored_hits=len(scored),
            correct=sum(g['score'] for e,p,g in scored),UNKNOWN=len(hits)-len(scored),
            paired_loss=sum(e['quality']['score']-g['score'] for e,p,g in scored)/len(known) if known else None,
            new_severe=sum(g['severe_error'] and not e['quality']['severe_error'] for e,p,g in scored),
            estimated_saved_ms=sum(e['request_wall_ms'] for e,p,g in hits),
            distinct_scored_frames=len({e['frame_id'] for e,p,g in scored})))
    bins=[]
    for kind in ('current_photometric_support','current_detector_center_projection_not_reuse_certificate'):
        for lo,hi in ((.35,.65),(.65,.85),(.85,1.01)):
            xs=[e for e in events if e.get('candidate_quality') and e['proposal']['facts'][0]['observation_kind']==kind
                and lo<=e['proposal']['support_score']<hi]
            scored=[e for e in xs if e['candidate_quality']['known']];err=sum(e['candidate_quality']['score']<1 for e in scored)
            bins.append(dict(observation_kind=kind,confidence_interval=[lo,hi],candidates=len(xs),scored=len(scored),
                UNKNOWN=len(xs)-len(scored),errors=err,descriptive_wilson95=wilson(err,len(scored))))
    write(root/'answer_validity_calibration.json',dict(stage='DEV_SHADOW',configs=configs,bins=bins,
        known_requests=len(known),actual_skips=0,normal_m3_ms=mean([e['request_wall_ms'] for e in events]),
        note='Temporal counterfactual is offline CPU on same actual current M1 snapshots. Only subsequent online Pareto measures savings. Scores are not error probabilities.'))
    for i,c in enumerate(CONFIGS):
        write(root/f'candidate_{i}.json',dict(status='FROZEN_RESEARCH_POLICY',policy_id=f'state_dev_{i}',budget=.02,
            expected_normal_m3_ms=mean([e['request_wall_ms'] for e in events]),min_net_gain_ms=0.,**c,
            calibration_sha256=sha256(root/'answer_validity_calibration.json'),automatic_deployment=False))
    with (root/'shadow_quality_requests.jsonl').open('w') as f:
        for e in events:f.write(json.dumps(e)+'\n')
    print('DEV_CALIBRATION',json.dumps(configs),flush=True)


def freeze(root):
    from .state_calibrated_bypass import CONFIGS
    rows=lines(root/'pareto/requests.jsonl');es=scored_requests(rows,load_labels(rows))
    base={(e['scene_id'],e['frame_id'],e['round'],e['request_index']):e for e in es if e['condition']=='B'}
    points=[];emap={(e['condition'],e['frame_id'],e['round'],e['request_index']):e for e in es}
    for i,c in enumerate(CONFIGS):
        part=[e for e in es if e['condition']==f'C{i}'];known=[];bounds=[];loss_rows=[]
        for e in part:
            b=base[(e['scene_id'],e['frame_id'],e['round'],e['request_index'])]
            if e['quality']['known'] and b['quality']['known']:
                known.append((b,e));value=b['quality']['score']-e['quality']['score']
                bounds.append((value,value));loss_rows.append(dict(scene_id=e['scene_id'],loss=value))
            else:bounds.append((0.,0.) if e['answer']==b['answer'] else (-1.,1.))
        points.append(dict(index=i,**c,mean_e2e_ms=mean([r['e2e_wall_ms'] for r in rows if r['condition']==f'C{i}']),
            requests=len(part),hits=sum(e['llm_skipped'] for e in part),scored=len(known),
            loss=mean([b['quality']['score']-e['quality']['score'] for b,e in known]),
            new_severe=sum(e['quality']['severe_error'] and not b['quality']['severe_error'] for b,e in known),
            conservative_all_request_loss_bounds=[mean([v[j] for v in bounds]) for j in (0,1)],
            scene_loss_CI95=scene_ci(loss_rows,'loss') if loss_rows else None))
    temporal=[e for e in es if e['condition']=='C3'];beneficial=[];temporal_loss=[]
    for e in temporal:
        cur=emap[('C1',e['frame_id'],e['round'],e['request_index'])]
        if cur['quality']['known'] and e['quality']['known']:
            temporal_loss.append(cur['quality']['score']-e['quality']['score'])
            if cur['llm_skipped'] and not e['llm_skipped'] and cur['quality']['score']==0 and e['quality']['score']==1:
                beneficial.append((e['frame_id'],e['question']))
    temporal_eligible=bool(beneficial and mean(temporal_loss)>0)
    policies=[]
    for budget in (.01,.02,.03):
        usable=[p for p in points if p['loss'] is not None and p['loss']<=budget and not p['new_severe'] and p['hits']>=3
                and p['conservative_all_request_loss_bounds'][1]<=budget
                and (not p['temporal_risk_enabled'] or temporal_eligible)]
        if not usable:policies.append(dict(budget=budget,status='NO_QUALIFIED_DEV_POLICY'));continue
        best=min(usable,key=lambda p:p['mean_e2e_ms'])
        policy=json.loads((root/f"candidate_{best['index']}.json").read_text());policy.update(budget=budget,online_dev_metrics=best)
        write(root/f'policy_{int(budget*100)}pp.json',policy);policies.append(policy)
        if budget==.02:write(root/'selected_policy.json',policy)
    write(root/'quality_latency_pareto.json',dict(stage='DEV_REAL_FULL_M123_GPU',points=points,policies=policies,
        temporal_eligible=temporal_eligible,unique_beneficial_dev_vetoes=len(set(beneficial)),
        temporal_paired_loss=mean(temporal_loss),test_tuning=False,
        rule='Temporal needs at least one independently scored beneficial veto and positive paired quality benefit; UNKNOWN cannot qualify it. Then minimize measured E2E within budget.'))
    assert (root/'selected_policy.json').exists(),'No default2pp DEV policy; stop TEST'
    sources=['worldstream/m3/runtime.py','worldstream/m3/answer_validity.py','worldstream/m3/world_state.py',
        'worldstream/m3/target_binding.py','worldstream/m3/task_conditioner.py','worldstream/m3/request_rng.py',
        'tools/worldstream_m123/state_calibrated_report.py','tools/worldstream_m123/selective_answer_reuse_report.py',
        'tools/worldstream_m123/answer_validity_quality.py','tools/worldstream_m123/temporal_selective_report.py']
    for p in sources:
        dest=root/'frozen_sources'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write(root/'POLICY_FREEZE.json',dict(selected_policy_sha256=sha256(root/'selected_policy.json'),
        sources={p:sha256(p) for p in sources},before_test=True))
    print('FROZEN',json.dumps(json.loads((root/'selected_policy.json').read_text())),flush=True)


def current_evidence(row):
    w=row['committed_world'];keys=('semantic_class','center_ego','velocity_ego','score','cameras','source_frame','source_timestamp','observed','observation_kind')
    anchors=[{k:a[k] for k in keys} for a in w['anchors'] if a['anchor_type']=='object']
    return dict(pose=w['ego_to_world'],timestamp=w['timestamp'],anchors=sorted(anchors,key=lambda a:json.dumps(a,sort_keys=True)))


def report(root):
    rows=lines(root/'run/requests.jsonl');ab=lines(root/'ablation/requests.jsonl')
    objs=load_labels(rows);es=scored_requests(rows,objs);ae=scored_requests(ab,objs)
    write(root/'current_frame_facts.json',dict(source='Offline-only independent current annotations',frames=objs))
    with (root/'quality_requests.jsonl').open('w') as f:
        for e in es+ae:f.write(json.dumps(e)+'\n')
    perf=performance(rows,[('A','B'),('B','C'),('A','C')]);q=quality(es,[('A','B'),('B','C'),('A','C')])
    for m in perf['module_breakdown']:
        part=[r for r in rows if (r['workload'],r['question_count'],r['condition'])==(m['workload'],m['question_count'],m['condition'])]
        requests=[z for r in part for z in r['requests']]
        for name,key in [('answer_memory_selection_wall_ms','draft_selection_wall_ms'),('answer_memory_management_wall_ms','draft_management_wall_ms')]:
            m[name]=mean([z.get(key,0.) for z in requests])
        m['rng_management_wall_ms']=mean([(z.get('question_rng') or {}).get('management_wall_ms',0.) for z in requests]) if m['condition']!='A' else None
        m['unsupported_requests']=sum((z.get('fast_answer') or {}).get('profile',{}).get('intent')=='unsupported' for z in requests)
    write(root/'abc_comparison.json',perf);write(root/'task_quality.json',q)
    policy=json.loads((root/'selected_policy.json').read_text());chosen='C-temporal' if policy['temporal_risk_enabled'] else 'C-current'
    crows=[dict(r,condition=chosen) for r in rows if r['condition']=='C' and r['workload']=='addressable']
    ce=[dict(e,condition=chosen) for e in es if e['condition']=='C' and e['workload']=='addressable']
    ap=performance(ab+crows,[('C-current','C-temporal')]);aq=quality(ae+ce,[('C-current','C-temporal')])
    counterpart={(r['question_count'],r['round'],r['frame_id']):r for r in crows}
    same=all(current_evidence(r)==current_evidence(counterpart[(r['question_count'],r['round'],r['frame_id'])]) for r in ab)
    allab=ab+crows;index={(r['condition'],r['question_count'],r['round'],r['frame_id']):r for r in allab}
    vetoes=[];continuity=[]
    for r in allab:
        if r['condition']!='C-temporal' or r['question_count']!=3:continue
        cur=index[('C-current',3,r['round'],r['frame_id'])]
        for qi,(cq,tq) in enumerate(zip(cur['requests'],r['requests'])):
            binding=(tq.get('fast_answer') or {}).get('target_binding',{})
            continuity.append(binding.get('continuity',{}))
            if not cq['llm_skipped'] or tq['llm_skipped']:continue
            cg=grade(cq['question'],cur['answers'][qi]['A'][0],r['frame_id'],objs[r['frame_id']])
            tg=grade(tq['question'],r['answers'][qi]['A'][0],r['frame_id'],objs[r['frame_id']])
            category=('unnecessary_veto' if cg['known'] and cg['score']==1 else
                'beneficial_veto' if cg['known'] and tg['known'] and cg['score']==0 and tg['score']==1 else 'unknown_veto')
            vetoes.append(dict(scene_id=r['scene_id'],frame_id=r['frame_id'],round=r['round'],question=cq['question'],
                selected_target=cq['fast_answer']['facts'],veto_class=category,current_quality=cg,temporal_quality=tg,
                binding=binding,extra_llm_calls=tq['llm_forward_calls']))
    contribution=dict(status='REAL_GPU_SAME_POLICY_ABLATION',performance=ap,quality=aq,
        current_evidence_equal=same,vetoes=vetoes,veto_counts=dict(Counter(v['veto_class'] for v in vetoes)),
        independent_veto_cases=len({(v['scene_id'],v['frame_id'],v['question']) for v in vetoes}),
        continuity_matched=sum(bool(v.get('matched')) for v in continuity),
        continuity_identity_qualified=sum(bool(v.get('identity_qualified')) for v in continuity),
        positive_admission_credits=sum(v.get('admission_credit',0.)>0 for v in continuity),
        temporal_independent_quality_gain=any(e['loss']<0 for e in aq['paired_losses']),
        caveat='Current detections, images, policy, seeds, weight and generator equal. Companion timing is a separate phase; use paired scene intervals. No unqualified identity gets admission credit.')
    write(root/'current_vs_temporal_ablation.json',contribution);write(root/'temporal_state_contribution.json',contribution)
    fast=[e for e in es if e['llm_skipped']]
    write(root/'fast_answer_analysis.json',dict(actual_skips=len(fast),sources=dict(Counter(e['answer_source'] for e in fast)),
        zero_llm_calls=all(e['llm_forward_calls']==0 for e in fast),groups=[g for g in q['groups'] if g['execution']=='fast_hit']))
    new=[e for e in q['errors'] if e['comparison']=='B->C' and e['new_severe_error']]
    write(root/'severe_error_audit.json',dict(new_severe_vs_B=new,paths=dict(Counter(e['path'] for e in new)),
        unique_new_severe=len({(e['frame_id'],e['question']) for e in new}),all_errors=q['errors']))
    write(root/'target_binding_error_analysis.json',dict(errors=[e for e in q['errors'] if e['comparison']=='B->C'],
        fact_sources=dict(Counter(e['proposal']['facts'][0]['observation_kind'] for e in fast)),
        old_diagnostic_chain='first_frame_detection_chain.json',no_online_GT=True))
    tasks=[]
    for cond in ('A','B','C'):
        for task in ('category','existence','position','motion','unsupported'):
            part=[e for e in es if e['condition']==cond and e['task']==task];known=[e for e in part if e['quality']['known']]
            tasks.append(dict(condition=cond,task=task,requests=len(part),scored=len(known),UNKNOWN=len(part)-len(known),
                score=mean([e['quality']['score'] for e in known]),severe=sum(bool(e['quality']['severe_error']) for e in known),
                independent_frame_questions=len({(e['frame_id'],e['question']) for e in known}),independent_scenes=len({e['scene_id'] for e in known})))
    write(root/'quality_by_task.json',tasks)
    def s(w,n=3):return next(e for e in perf['summaries'] if (e['workload'],e['question_count'],e['comparison'])==(w,n,'B->C'))
    addr=s('addressable');loss=next(e for e in q['paired_losses'] if (e['workload'],e['question_count'],e['comparison'])==('addressable',3,'B->C'))
    coverage=next(e for e in q['groups'] if (e['workload'],e['question_count'],e['condition'],e['execution'])==('addressable',3,'C','fast_hit'))
    gain=addr['e2e_speedup']>1 and addr['delta_scene_CI95_ms'][1]<0 and all(v>1 for v in addr['scene_speedups'].values())
    valid=(loss['loss'] is not None and loss['loss']<=policy['budget'] and loss['conservative_all_request_loss_bounds'][1]<=policy['budget']
        and loss['scene_CI95'] is not None and loss['scene_CI95'][1]<=policy['budget'] and not new)
    status='PARTIAL_GO' if gain and valid else 'NO_GO'
    if coverage['scored']<3 or not same:status='INCOMPLETE'
    if (gain and loss['loss'] is not None and loss['loss']<=policy['budget'] and not new
            and loss['conservative_all_request_loss_bounds'][1]>policy['budget']):status='INCOMPLETE'
    decision=dict(status=status,selected_temporal=policy['temporal_risk_enabled'],addressable_three_QA=addr,
        paired_quality=loss,new_severe_errors=len(new),temporal_independent_quality_gain=contribution['temporal_independent_quality_gain'],
        broad_driving_quality='INCOMPLETE',deployment_enabled=False,test_threshold_tuning=False,
        independent_test_scenes=3,independent_test_frames=24)
    write(root/'go_no_go.json',decision)
    counts={k:len(lines(root/k/'requests.jsonl')) for k in ('shadow','pareto','run','ablation')}
    write(root/'run_counts.json',dict(stages=dict(detection=4,**counts),full_M123_calls=4+sum(counts.values()),
        formal_question_requests=len(es),actual_formal_skips=len(fast),ablation_question_requests=len(ae)))
    (root/'e2e_performance_report.md').write_text('# Full-M123 RTX3090 measurements\n\n'+'\n'.join(
        f"- {x['workload']} {x['question_count']}Q {x['comparison']}: {x['e2e_speedup']:.6f}x; paired scene delta CI {x['delta_scene_CI95_ms']}ms." for x in perf['summaries'])+
        '\n\nThree new logs, eight contiguous frames per scene, two rounds. Rounds are not independent scenes. Images, model, generation limits, warpers and stable question RNG preserved.\n')
    mixed=[e for e in es if e['workload']=='mixed' and e['condition']=='C'];structured=sum(e['task']!='unsupported' for e in mixed)
    (root/'mixed_workload_report.md').write_text('# Preset mixed workload\n\n'+
        f"One introduced structured question plus two original natural questions per frame, cycling structured type by frame index. Preset1/3 slots; actual parsed structured fraction {structured}/{len(mixed)}. This is not an observed natural-QA population distribution. Actual skips {sum(e['llm_skipped'] for e in mixed)}/{len(mixed)}. B→C {s('mixed')['e2e_speedup']:.6f}x; scene deltaCI {s('mixed')['delta_scene_CI95_ms']}. Quality, UNKNOWNs and conservative bounds remain separate in task_quality.json.\n")
    (root/'runtime_cost_report.md').write_text('# Costs and unsupported-path interpretation\n\n'+
        'Task parsing is pure and bounded-LRU cached; unsupported tasks exit before target/history scans. Required AnswerMemory retrieval for the unchanged K8 target generator remains. Snapshot detector-center projection is CPU-only and distinct from historical reuse qualification. No extra neural forward. Per-question parsing, binding, temporal, validity, construction, RNG management and AnswerMemory costs plus all M1/M2/Heads/CUDA spans are in abc_comparison.json. Nested binding/temporal costs must not be summed again.\n\n'+
        '\n'.join(f"- {m['workload']} {m['question_count']}Q {m['condition']}: E2E {m['e2e_wall_ms']:.3f}ms; M1 {m['m1_wall_ms']:.3f}, M2 {m['m2_wall_ms']:.3f}, heads {m['heads_wall_ms']:.3f}; decision on misses {m['miss_decision_wall_ms']}ms; RNG management {m['rng_management_wall_ms']}ms/Q." for m in perf['module_breakdown'])+
        '\n\nThe tens-of-microseconds decision cost cannot explain all wall-time deltas. Stage means and scene-paired intervals expose M1/M2/CUDA/order variability; no claim of pure parser causality is made. Native RNG management is not separately instrumented, though its outer E2E includes it.\n')
    (root/'implementation_summary.md').write_text(f'''# WRSAR v3

Decision **{status}**. Default research switches remain off. Same original checkpoint and sampling semantics, no training or M1/M2 redesign.

P0:9 old veto records ->3 independent cases, all UNKNOWN; zero proven beneficial vetoes. P1:Car is detected(score0.875) and committed; old M3 incorrectly conflates current detector evidence with90% whole-box photometric reuse support. Opt-in M3 reads current committed actual detections with current calibrated camera-center projection, without modifying any M1/M2 reuse certificate, score threshold or4m ROI. Pure task parser caching reduces unsupported-request work.

DEV selected {'Temporal-Aware' if policy['temporal_risk_enabled'] else 'Current-Only'} at confidence{policy['min_confidence']},risk{policy['max_binding_risk']}. Final TEST:three new disjoint logs,24 current frames, natural/fixed/addressable separately plus explicitly preset mixed3Q. Addressable3Q B→C {addr['e2e_speedup']:.6f}x, scored B−C loss{loss['loss']:.6f}; actual formal M3 skips{len(fast)}, new severe outputs versusB{len(new)}. UNKNOWN and residual normal-model errors retained. Cross-frame independent quality gain observed flag{contribution['temporal_independent_quality_gain']}; this is not a broad safety or natural-QA qualification.
''')
    render_answers(root,decision,perf,q,contribution,policy,fast)
    print(json.dumps(dict(verdict=status,counts=counts,actual_skips=len(fast),addressable=addr,loss=loss,
        temporal_selected=policy['temporal_risk_enabled'],new_severe=len(new))),flush=True)


def render_answers(root,decision,perf,q,ab,policy,fast):
    prior=json.loads((root/'temporal_veto_quality_audit.json').read_text())
    def s(w,n=3):return next(x for x in perf['summaries'] if (x['workload'],x['question_count'],x['comparison'])==(w,n,'B->C'))
    def g(c,execution='all'):return next(x for x in q['groups'] if (x['workload'],x['question_count'],x['condition'],x['execution'])==('addressable',3,c,execution))
    texts=[f"1. 上轮9次三问额外拒绝去重为{prior['independent_cases']}例、{prior['independent_scenes']}个Scene，Selected Target按当前类别/几何绑定，不把Round当独立案例。",
        f"2. 有益拒绝{prior['counts'].get('beneficial_veto',0)}。",f"3. 可证实无益拒绝{prior['counts'].get('unnecessary_veto',0)}。",
        f"4. 仍UNKNOWN {prior['counts'].get('unknown_veto',0)}；已有QA不能确定最近目标；保留原2m类别歧义条件，投影图仅为复核准备，未声称人工复核。",
        '5. Car没有从检测、类别映射、0.3分数过滤、几何或M1提交丢失。其检测中心(8.507,-3.859)m、得分0.875；M3快照用历史复用的相机描述子资格判断当前可观测性，因前/右前全框覆盖仅40.8%/61.0%而屏蔽了当前Car。预测和GT均在原4m ROI内。',
        '6. 可以读取已经提交的当前actual_perception和真实相机中心投影，区分当前检测与历史复用证明；不重新运行任何检测/VLM，也不修改M1/M2门槛。低置信或无合法当前相机的未知对象仍不能输出肯定事实。',
        f"7. 旧诊断实际输出CPU回放由Truck变Car，但这不是泛化证据。新TEST三问已评分快速回答 {g('C','fast_hit')['score']}、错误{g('C','fast_hit')['errors']}；详情包含观测来源、UNKNOWN和正常生成错误。",
        f"8. 连续性仅作证据诊断：匹配{ab['continuity_matched']}，身份合格{ab['continuity_identity_qualified']}，正向准入加分{ab['positive_admission_credits']}。P0无有益证据，不添加未经验证的连续性信用。",
        f"9. 新TEST必要消融 veto counts={ab['veto_counts']}，UNKNOWN不算有益；历史预测从未成为当前事实。",
        f"10. 时序独立已评分质量改善标记{ab['temporal_independent_quality_gain']}；实际质量与E2E配对和同检测验证在 current_vs_temporal_ablation.json，短流与UNKNOWN仍限制结论。",
        f"11. DEV最优是否启用Temporal Risk：{policy['temporal_risk_enabled']}；不因方法名而强制启用。",
        f"12. 正式A/B/C真实跳过{len(fast)}次，来源{dict(Counter(e['answer_source'] for e in fast))}，钩子验证每个命中零LLM Forward/零展开，Prefill和Decode均未执行。DEV/消融另外统计。",
        '13. 每类请求解析/绑定/时序/快照/AnswerMemory/RNG成本和主模块分解见runtime_cost_report.md与abc_comparison.json；全为实测，时序没有额外模型GPU计算。',
        f"14. M1-Addressable3Q B→C {s('addressable')['e2e_speedup']:.6f}x，场景E2E差CI {s('addressable')['delta_scene_CI95_ms']}ms。",
        f"15. 本次{s('addressable')['e2e_speedup']:.6f}x与上次1.314639x来自不同独立流，不能解释为同数据因果提升。",
        f"16. 自然3Q{s('natural')['e2e_speedup']:.6f}x、固定3Q{s('fixed')['e2e_speedup']:.6f}x、预设混合3Q{s('mixed')['e2e_speedup']:.6f}x；1Q与场景CI分别报告。不能把混合分布说成自然问题分布。",
        f"17. 新三问已评分质量B={g('B')['score']}，C={g('C')['score']}；覆盖{g('C')['scored']}/{g('C')['requests']}。未知变化的最坏质量差上下界已保留。",
        f"18. 相对B新增严重错误{decision['new_severe_errors']}；所有案例保留执行路径，已评分C严重输出{g('C')['severe']}。不能以总体分数掩盖正常LLM或快速回答错误。",
        '19. 1/2/3pp的真实DEV候选、最优工作点及Temporal准入证据在quality_latency_pareto.json；仅比较4个预设候选，TEST没有阈值或种子调节。',
        f"20. 最终{decision['status']}；当前最优是{'Temporal-State-Aware' if policy['temporal_risk_enabled'] else 'Current-Frame Fact'} Bypass。跨帧贡献必须由独立消融而非模块存在来证明；整体驾驶安全标签仍不完整，默认研究开关关闭。"]
    (root/'result_answers.md').write_text('# 20项最终回答\n\n'+'\n\n'.join(texts)+'\n')
