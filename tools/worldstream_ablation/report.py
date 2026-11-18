"""Offline GT evaluation and paired compute denominators; no inference code here."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import pickle
import re
import numpy as np
from .registry import GROUPS, VARIANTS


def target_signature(question,objects):
    """GT answer semantics, deliberately independent of model/cached answers."""
    from worldstream.m3.task_conditioner import question_facts
    from worldstream.m3.answer_validity import fact_region
    from worldstream.m3.world_delta import VEHICLES
    from tools.worldstream_m123.fact_bypass_identity import question_target
    p = question_facts(question)
    if p.intent == 'unsupported' or p.classes == ('object',):
        return None
    classes = set(p.classes)
    if 'vehicle' in classes:
        classes = (classes-{'vehicle'})|VEHICLES
    region = [g for g in objects if g['semantic_class'] in classes and fact_region(g['center_ego'],p)]
    visible = [g for g in region if g['visible']]
    if not visible:
        # Visible-target benchmark: a current annotated but partially visible
        # candidate is ambiguous, not labelled absent. No visible/in-view target
        # in official GT defines absence for the closed question grammar.
        if any(g['cameras'] for g in region):
            return None
        return (p.intent,'absent')
    target = question_target(question,objects)
    if not target.get('answer_known'):
        return None
    if p.intent == 'existence':
        return (p.intent,'present')
    if p.intent == 'category':
        return (p.intent,target['semantic_class'])
    if p.intent == 'position':
        return (p.intent,tuple(target['directions']))
    return None


def grade_answer(question,answer,token,objects):
    from tools.worldstream_m123.continuous_stream_report import grade
    signature = target_signature(question,objects[token])
    if signature is not None and signature[-1] == 'absent':
        from worldstream.m3.task_conditioner import question_facts
        from worldstream.m3.world_delta import VEHICLES
        from tools.worldstream_m123.selective_answer_reuse_report import extracted_classes
        text = ' '.join(answer.lower().split())
        # Negation must refer to the *queried* object. "No pedestrians, a car
        # is ahead" cannot answer a vehicle-absence question correctly.
        uncertain = bool(re.search(r"\b(uncertain|unknown|cannot|can't|unable|maybe|might)\b",text))
        classes = set(question_facts(question).classes)
        nouns = dict(vehicle=r'vehicles?',car=r'cars?|sedans?|suvs?|vans?|taxis?',
            truck=r'trucks?',bus=r'bus|buses',pedestrian=r'pedestrians?|people|person',
            bicycle=r'bicycles?|cyclists?',motorcycle=r'motorcycles?',trailer=r'trailers?',
            construction_vehicle=r'construction vehicles?')
        noun = '|'.join(nouns.get(c,re.escape(c)) for c in sorted(classes))
        if classes and classes <= VEHICLES:
            # "No vehicles" also entails "no car/truck/...".
            noun += '|vehicles?'
        qualifiers = r'(?:(?:visible|nearby|other|any|a|an|single)\s+)*'
        pattern = rf"\b(?:(?:no|without)\s+{qualifiers}|(?:is not|are not|isn't|aren't)\s+{qualifiers})(?:{noun})\b"
        matches = list(re.finditer(pattern,text))
        bare = bool(re.fullmatch(r'(?:no|none|no one)[.! ]*',text))
        negative = bare or bool(matches)
        remainder = re.sub(pattern,' ',text)
        target_classes = (classes-{'vehicle'})|VEHICLES if 'vehicle' in classes else classes
        contradictory = bool(re.search(r'\byes\b',text) or extracted_classes(remainder)&target_classes)
        return dict(known=True,score=float(negative and not uncertain and not contradictory),
                    reason='official_GT_no_visible_target',absence_label=True)
    return grade(question,answer,token,objects)


def scenario_tags(previous,token,objects,infos):
    if previous is None:
        return ['cold_start']
    a,b = infos[previous],infos[token]
    tags = []
    ego_shift = np.linalg.norm(np.asarray(b['ego2global_translation'])-a['ego2global_translation'])
    from pyquaternion import Quaternion
    rotation_a = Quaternion(a['ego2global_rotation']).rotation_matrix
    rotation_b = Quaternion(b['ego2global_rotation']).rotation_matrix
    angle = math.acos(float(np.clip((np.trace(rotation_a.T@rotation_b)-1)/2,-1,1)))
    if ego_shift>.5 or angle>math.radians(2):
        tags.append('ego_motion')
    old = {g['instance_id']:g for g in objects[previous]}
    now = {g['instance_id']:g for g in objects[token]}
    for k,g in old.items():
        h = now.get(k)
        if g['visible'] and (h is None or not h['cameras']):
            tags.append('object_disappearance')
        elif g['visible'] and h is not None and not h['visible']:
            tags.append('local_occlusion')
        if h is not None and g['semantic_class'] in ('car','truck','bus','trailer','motorcycle','bicycle'):
            p = rotation_a@np.asarray(g['center_ego'])+a['ego2global_translation']
            q = rotation_b@np.asarray(h['center_ego'])+b['ego2global_translation']
            if np.linalg.norm(p-q)>.5:
                tags.append('vehicle_motion')
    return sorted(set(tags)) or ['stable_other']


def summarize_events(events):
    factual = [e for e in events if e['signature'] is not None]
    scored = [e for e in factual if e['quality']['known']]
    changes = [e for e in events if e['change_opportunity']]
    reused_changes = [e for e in events if e.get('cache_source_changed')]
    return dict(Factual_QA_Accuracy_percent=100*sum(e['quality']['score']==1 for e in scored)/len(scored) if scored else None,
        factual_scored=len(scored),questions=len(events),factual_unscored=len(events)-len(scored),
        unknown_reasons=dict(Counter(e['quality'].get('reason','unscorable') for e in events if not e['quality']['known'])),
        Stale_Answer_Rate_percent=100*sum(e['stale'] for e in changes)/len(changes) if changes else None,
        stale_answers=sum(e['stale'] for e in changes),changed_query_denominator=len(changes),
        stale_on_reused_changed_source_percent=100*sum(e['cache_source_stale'] for e in reused_changes)/len(reused_changes) if reused_changes else None,
        reused_changed_source_denominator=len(reused_changes),
        original_QA_limited_fact_score_percent=(100*np.mean([e['quality']['score'] for e in events
            if e['group']=='other' and e['quality']['known']]) if any(e['group']=='other' and e['quality']['known'] for e in events) else None))


def summarize(rows,events,baseline):
    def avg(values):
        return float(np.mean(values)) if values else None
    timing = lambda label:[r['timings'].get(label,{}).get('cuda_ms',0.) for r in rows]
    visual = timing('visual')
    measured = {label:sum(r['executed_macs'][label] for r in rows) for label in ('visual','llm')}
    native = {label:sum(r['executed_macs'][label] for r in baseline) for label in ('visual','llm')}
    requests = [q for row in rows for q in row['requests']]
    proposed = sum(q.get('draft_candidate_tokens',0) for q in requests)
    accepted = sum(q.get('draft_accepted_tokens',0) for q in requests)
    nqueries = sum(len(r['questions']) for r in rows)
    return dict(**summarize_events(events),frames=len(rows),
        OGSM_Latency_ms=avg([r['ogsm_precheck_commit_ms'] for r in rows]),
        Visual_Backbone_Latency_ms=avg(visual),P95_Visual_Backbone_Latency_ms=float(np.percentile(visual,95)),
        E2E_Latency_ms=avg([r['e2e_ms'] for r in rows]),
        LLM_Prefill_Latency_ms=sum(timing('prefill'))/nqueries if nqueries else None,
        LLM_Decode_Latency_ms=sum(timing('decode'))/nqueries if nqueries else None,
        Visual_Compute_Reduction_percent=100*(1-measured['visual']/native['visual']) if native['visual'] else None,
        LLM_Compute_Reduction_percent=100*(1-measured['llm']/native['llm']) if native['llm'] else None,
        actual_visual_MACs=measured['visual'],actual_LLM_MACs=measured['llm'],
        native_visual_MACs=native['visual'],native_LLM_MACs=native['llm'],
        Direct_Answer_Rate_percent=100*sum(q.get('direct_answer',q.get('executor')=='QuestionReferencedFactReuse') for q in requests)/nqueries if nqueries else None,
        Candidate_Token_Acceptance_Rate_percent=100*accepted/proposed if proposed else None,
        proposed_candidate_tokens=proposed,accepted_candidate_tokens=accepted,
        prefill_decode_latency_denominator='all queries, direct/cache-hit routes contribute zero; CUDA forward spans only',
        compute_ratio='sum actually executed dense MACs / Native on identical frames/questions; output-length differences retained',
        latency='Visual CUDA span includes DAVP scheduling/cache overhead; E2E synchronized host wall incl preprocessing/export; LLM Prefill/Decode are sums of forward CUDA spans, not Python token-selection time')


def report(out):
    from .run import read,write,sha
    from tools.worldstream_m123.answer_validity_quality import load_current_annotations
    out = Path(out)
    manifest = read(out/'manifest.json')
    for label in ('source_hashes','data_hashes'):
        for path,digest in manifest[label].items():
            if sha(path)!=digest:
                raise RuntimeError('Source/data changed since preparation:'+path)
    tokens = [t for s in manifest['scenes'] for t in s['tokens']]
    objects = load_current_annotations(manifest['infos'],set(tokens),'data/nuscenes')
    infos = {i['token']:i for i in pickle.load(Path(manifest['infos']).open('rb'))['infos'] if i['token'] in set(tokens)}
    rows = {}
    for name in manifest['variants']:
        directory = out/name
        completion = read(directory/'inference_complete.json')
        if completion['status']!='COMPLETED' or completion['receipts_sha256']!=sha(directory/'receipts.jsonl'):
            raise RuntimeError('Incomplete/tampered inference:'+name)
        rows[name] = [json.loads(line) for line in (directory/'receipts.jsonl').read_text().splitlines()]
        if [r['frame_id'] for r in rows[name]]!=tokens or len(rows[name])!=completion['frames']:
            raise RuntimeError('Missing/duplicated/reordered frames:'+name)
        for native,row in zip(rows.get('native',rows[name]),rows[name]):
            if native['questions']!=row['questions']:
                raise RuntimeError('Unpaired workloads:'+name)
    metrics = {}
    tags = {}
    for scene in manifest['scenes']:
        previous = None
        for token in scene['tokens']:
            tags[token] = scenario_tags(previous,token,objects,infos)
            previous = token
    write(out/'scenario_tags.json',tags)
    for name,part in rows.items():
        events,history = [],{}
        for row in part:
            for qi,(question,answer,group) in enumerate(zip(row['questions'],row['answers'],row['groups'])):
                token = row['frame_id']
                key = (row['scene_id'],question)
                signature = target_signature(question,objects[token])
                quality = grade_answer(question,answer,token,objects)
                prev = history.get(key)
                change = bool(prev and prev['signature'] is not None and signature is not None and prev['signature']!=signature)
                stale = bool(change and quality['known'] and quality['score']<1 and
                    grade_answer(question,answer,prev['token'],objects).get('score')==1)
                route = row['requests'][qi] if row['requests'] else {}
                source = route.get('source_frame')
                source_sig = target_signature(question,objects[source]) if source in objects else None
                source_changed = bool(source_sig is not None and signature is not None and source_sig!=signature)
                event = dict(variant=name,scene_id=row['scene_id'],frame_id=token,question=question,answer=answer,
                    group=group,signature=signature,quality=quality,change_opportunity=change,stale=stale,
                    previous_frame=prev['token'] if prev else None,scenario_tags=tags[token],
                    cache_source_frame=source,cache_source_changed=source_changed,
                    cache_source_stale=bool(source_changed and quality['known'] and quality['score']<1 and
                        grade_answer(question,answer,source,objects).get('score')==1))
                events.append(event)
                history[key] = dict(token=token,signature=signature)
        metrics[name] = summarize(part,events,rows['native'])
        metrics[name]['scenarios'] = {tag:summarize_events([e for e in events if tag in e['scenario_tags']])
            for tag in sorted({t for ts in tags.values() for t in ts})}
        with (out/name/'per_question.jsonl').open('w') as f:
            for event in events:
                f.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        write(out/name/'metrics.json',metrics[name])
    write(out/'metrics.json',dict(smoke=manifest['smoke'],metrics=metrics,definitions={
        k:manifest[k] for k in ('compute_definition','accuracy_definition','stale_definition')}))
    fields = ['variant','label','Factual_QA_Accuracy_percent','Stale_Answer_Rate_percent','factual_scored',
        'changed_query_denominator','OGSM_Latency_ms','Visual_Backbone_Latency_ms','P95_Visual_Backbone_Latency_ms',
        'Visual_Compute_Reduction_percent','LLM_Prefill_Latency_ms','LLM_Decode_Latency_ms','LLM_Compute_Reduction_percent',
        'E2E_Latency_ms','Direct_Answer_Rate_percent','Candidate_Token_Acceptance_Rate_percent']
    lines = ['# WorldStream 消融实验'+('（SMOKE：仅启动验证，不作论文结果）' if manifest['smoke'] else ''),'',
        f"{len(manifest['scenes'])} Scene / {len(tokens)} frames; {len(metrics)} unique variants; Native配对计算量参照。",'',
        '事实问答为可评分闭合事实语法的完全正确比例；UNKNOWN、变化机会分母和场景覆盖均独立记录。',
        'Stale为已知GT语义变化后仍回答前次事实的行为比例，不把所有错误归因于缓存。零变化机会为N/A。',
        '时延包含冷启动帧。Visual为CUDA流跨度（含调度/缓存主机空隙）；LLM为实际Forward的CUDA跨度，按全部查询平均，缓存/直答为零。',
        '计算量为实际执行Linear/Conv/Attention的形状推导MACs；包含完整Patch Embed、全局Suffix与LoRA，排除逐元素算子/拷贝。不是窗口复用率，也不是硬件计数器FLOPs。','']
    columns = dict(module=['Factual_QA_Accuracy_percent'],
        ogsm=['Factual_QA_Accuracy_percent','Stale_Answer_Rate_percent','OGSM_Latency_ms','E2E_Latency_ms'],
        davp=['Visual_Backbone_Latency_ms','Visual_Compute_Reduction_percent','Factual_QA_Accuracy_percent',
              'E2E_Latency_ms','P95_Visual_Backbone_Latency_ms'],
        qcse=['Factual_QA_Accuracy_percent','Stale_Answer_Rate_percent','LLM_Prefill_Latency_ms',
              'LLM_Decode_Latency_ms','E2E_Latency_ms','LLM_Compute_Reduction_percent'])
    show = lambda v:'N/A' if v is None else f'{v:.3f}'
    for group,names in GROUPS.items():
        available = [n for n in names if n in metrics]
        if len(available)<2:
            continue
        lines += [f'## {group}', '', '|配置|'+'|'.join(columns[group])+'|','|---|'+'|'.join(['---:']*len(columns[group]))+'|']
        for name in available:
            lines.append('|'+VARIANTS[name].label+'|'+'|'.join(show(metrics[name][c]) for c in columns[group])+'|')
        lines.append('')
        with (out/(group+'.csv')).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for name in available:
                writer.writerow(dict(variant=name,label=VARIANTS[name].label,
                    **{k:metrics[name].get(k) for k in fields if k not in ('variant','label')}))
    lines += ['## 场景覆盖','', '|场景|帧数|','|---|---:|']
    for tag,count in sorted(Counter(t for ts in tags.values() for t in ts).items()):
        lines.append(f'|{tag}|{count}|')
    lines += ['', '场景标签来自离线GT，彼此可重叠，未参与推理或样本挑选。各配置的分场景Accuracy/Stale与分母见metrics.json。',
        'Full R0当前采用current_only事实直答；历史状态验证主要通过DAVP/候选执行影响QA。替代状态的历史有效性被显式记录，不能把QA差异单独解释为状态有效性证明。']
    (out/'ABLATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'completion.json',dict(status='COMPLETED',smoke=manifest['smoke'],variants=list(metrics),
        frames_per_variant=len(tokens),report='ABLATION_REPORT.md'))
