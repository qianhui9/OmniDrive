"""R0-required utilities; historical alternative runners have been retired."""

from collections import Counter, defaultdict

import math

from .fact_bypass_identity import match_instances

def raw_anchors(row, floor=.3):
    from worldstream.m1.omnidrive_adapter import CLASS_NAMES
    raw=row['visual_quality']
    result=[]
    for i,(box,cls,score) in enumerate(zip(raw['boxes'],raw['labels'],raw['scores'])):
        if score<floor:continue
        center=list(box[:3]);center[2]+=box[5]/2
        result.append(dict(anchor_id=str(i),anchor_type='object',observed=True,
            source_frame=row['frame_id'],source_timestamp=row['timestamp'],
            evidence_origin='actual_perception',center_ego=center,
            semantic_class=CLASS_NAMES[cls],score=score,cameras=[]))
    return result


def current_anchors(row):
    return [a for a in (row.get('committed_world') or {}).get('anchors',[])
        if a['anchor_type']=='object' and a['observed'] and a['source_frame']==row['frame_id']
        and a['source_timestamp']==row['timestamp'] and a['evidence_origin']=='actual_perception']


def eligible(anchors,question):
    from worldstream.m3.task_conditioner import question_facts
    from worldstream.m3.answer_validity import fact_region
    from worldstream.m3.world_delta import VEHICLES
    p=question_facts(question);classes=set(p.classes)
    if 'vehicle' in classes:classes=(classes-{'vehicle'})|VEHICLES
    if 'object' in classes:classes=set()
    return sorted([a for a in anchors if a['cameras'] and a['score'] is not None
        and math.isfinite(a['score']) and a['score']>=.35
        and (not classes or a['semantic_class'] in classes)
        and all(math.isfinite(v) for v in a['center_ego']) and fact_region(a['center_ego'],p)],
        key=lambda a:(math.hypot(*a['center_ego'][:2]),-a['score'],a['anchor_id']))


def detection_frame(row, anchors, objects):
    """Explicit GT denominators, ambiguity-aware recall and precision bounds."""
    matches=match_instances(anchors,objects,row['frame_id'],row['timestamp'])
    correct={m['annotation_id'] for m in matches.values() if m['known'] and m['class_correct']}
    by_annotation={m['annotation_id']:m for m in matches.values() if m['known']}
    sets={
        'all_mapped_GT':objects,
        'visible_GT':[g for g in objects if g['visible']],
        'visible_within50m':[g for g in objects if g['visible'] and math.hypot(*g['center_ego'][:2])<=50.]}
    for cls in sorted({g['semantic_class'] for g in objects}):
        sets['class/'+cls]=[g for g in objects if g['visible'] and g['semantic_class']==cls]
    for low,high in ((0,10),(10,20),(20,30),(30,50),(50,math.inf)):
        sets[f'distance/{low}-{high}m']=[g for g in objects if g['visible'] and low<=math.hypot(*g['center_ego'][:2])<high]
    groups={}
    for name,gs in sets.items():
        tp=sum(g['annotation_id'] in correct for g in gs)
        fn=unknown=0
        for g in gs:
            if g['annotation_id'] in correct:continue
            potential=any(a['semantic_class']==g['semantic_class'] and math.dist(a['center_ego'],g['center_ego'])<2. for a in anchors)
            if potential:unknown+=1
            else:fn+=1
        groups[name]=dict(GT=len(gs),TP=tp,FN=fn,UNKNOWN_GT=unknown,
            localization_sum_m=sum(by_annotation[g['annotation_id']]['center_distance_m'] for g in gs if g['annotation_id'] in correct))
    known=[m for m in matches.values() if m['known']]
    return dict(groups=groups,predictions=len(anchors),matched=len(known),class_correct=sum(m['class_correct'] for m in known),
        confirmed_class_FP=sum(not m['class_correct'] for m in known),UNKNOWN_predictions=len(anchors)-len(known),matches=matches)


def aggregate_detection(frames):
    groups={}
    for name in sorted({k for r in frames for k in r['groups']}):
        c=Counter()
        for r in frames:c.update(r['groups'].get(name,{}))
        n=c['GT'];tp=c['TP'];u=c['UNKNOWN_GT']
        groups[name]=dict(c,proven_class_recall=tp/n if n else None,
            class_recall_bounds=[tp/n,(tp+u)/n] if n else None,
            mean_proven_localization_m=c['localization_sum_m']/tp if tp else None)
    totals={k:sum(r[k] for r in frames) for k in ('predictions','matched','class_correct','confirmed_class_FP','UNKNOWN_predictions')}
    n=totals['predictions'];t=totals['class_correct'];u=totals['UNKNOWN_predictions']
    return dict(groups=groups,**totals,FP=None,
        FP_status='NOT_CERTIFIED: unmatched/ambiguous detections are UNKNOWN, not assumed false positives',
        conditional_matched_class_precision=t/totals['matched'] if totals['matched'] else None,
        unconditional_class_precision_bounds=[t/n,(t+u)/n] if n else None)


def association(rows, objects):
    counts=Counter();lifetimes=[];recoveries=[];streams=defaultdict(list)
    for r in rows:streams[r['scene_id']].append(r)
    for scene,part in streams.items():
        previous={};previous_instances={};tracks=defaultdict(list);last_seen={};seen_at={}
        for r in sorted(part,key=lambda r:r['frame_index']):
            anchors=current_anchors(r)
            matches=match_instances(anchors,objects[r['frame_id']],r['frame_id'],r['timestamp'])
            current={aid:m['instance_id'] for aid,m in matches.items() if m['known']}
            visible={g['instance_id'] for g in objects[r['frame_id']] if g['visible']}
            for a,_ in r['visual_quality']['association_pairs']:
                if a not in matches:continue
                counts['accepted_links']+=1
                if a in previous and a in current:
                    counts['scored_links']+=1
                    counts['correct_links' if previous[a]==current[a] else 'wrong_links']+=1
                    if previous[a]==current[a] and current[a] in visible and current[a] in seen_at:
                        counts['correct_visible_links']+=1
                else:counts['UNKNOWN_links']+=1
            instances={v:k for k,v in current.items()}
            common=set(instances)&set(previous_instances)
            counts['consecutive_scored_GT_instances']+=len(common)
            counts['GT_instance_ID_changes']+=sum(instances[i]!=previous_instances[i] for i in common)
            if seen_at:
                counts['consecutive_visible_GT_opportunities']+=len(visible&set(seen_at))
            for inst,aid in instances.items():
                tracks[aid].append(r['timestamp'])
                if inst in last_seen and r['frame_index']>last_seen[inst][0]+1:
                    recoveries.append(dict(scene=scene,instance=inst,gap_frames=r['frame_index']-last_seen[inst][0]-1,
                        same_anchor=aid==last_seen[inst][1],frame_id=r['frame_id']))
                last_seen[inst]=(r['frame_index'],aid)
            previous=current;previous_instances=instances;seen_at={i:True for i in visible}
        lifetimes.extend(dict(scene_id=scene,anchor_id=aid,scored_observed_frames=len(ts),
            observed_span_s=max(ts)-min(ts)) for aid,ts in tracks.items())
    return dict(counts=dict(counts),conditional_link_precision=counts['correct_links']/counts['scored_links'] if counts['scored_links'] else None,
        effective_association_coverage=counts['correct_visible_links']/counts['consecutive_visible_GT_opportunities'] if counts['consecutive_visible_GT_opportunities'] else None,
        ID_change_rate=counts['GT_instance_ID_changes']/counts['consecutive_scored_GT_instances'] if counts['consecutive_scored_GT_instances'] else None,
        observed_retention=lifetimes,loss_recovery_events=recoveries,
        scope='Partial unique-GT matching. Observed spans are censored by 16-frame streams and UNKNOWN gaps, not true lifetime; not MOTA/IDF1/HOTA.')
