"""Offline-only nuScenes identity scoring. Never imported by online M1.

Unique class-compatible, sensor-supported 2m assignment (0.25m ambiguity
margin) defines scorable detection identities. This is not a pixel mask oracle.
Unknown proposals are diagnostics, not false accepted identities.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import numpy as np
from scipy.stats import beta
from .dataset import load_oracle,log_context_from_root
from .evidence_oracle import identity_population
from .scene_protocol import ratio,distribution
from .effective_acceptance import sha


def score_links(proposals, identities, scene, log):
    result=[]
    for p in proposals:
        source=identities.get(p['source_frame_id'],{}).get(p['anchor_id'])
        target=identities.get(p['current_frame_id'],{}).get(p.get('target_anchor_id'))
        status='unscorable' if source is None or target is None else 'correct' if source==target else 'wrong'
        result.append(dict(scene_id=scene,log_token=log,source_frame_id=p['source_frame_id'],
            current_frame_id=p['current_frame_id'],source_anchor_id=p['anchor_id'],target_anchor_id=p.get('target_anchor_id'),
            proposal_status=p['status'],formal_certified=bool(p.get('certified',False)),score=status,
            source_instance=source,target_instance=target,
            reason='unique_supported_GT_instances' if status!='unscorable' else 'source_or_target_not_uniquely_scorable'))
    return result


def summarize(rows):
    counts=Counter(total_proposals=len(rows),**{k:0 for k in ('correct','wrong','unscorable','candidates','candidate_correct',
        'candidate_wrong','candidate_unscorable','formally_certified','formal_wrong','formal_unscorable')})
    for x in rows:
        counts[x['score']]+=1
        if x['proposal_status']=='candidate':
            counts['candidates']+=1;counts['candidate_'+x['score']]+=1
        if x['formal_certified']:
            counts['formally_certified']+=1
            if x['score']=='wrong':counts['formal_wrong']+=1
            if x['score']=='unscorable':counts['formal_unscorable']+=1
    return dict(counts=dict(counts),scorable_coverage=ratio(counts['correct']+counts['wrong'],len(rows)),
        candidate_coverage=ratio(counts['candidates'],len(rows)),
        candidate_accuracy=ratio(counts['candidate_correct'],counts['candidate_correct']+counts['candidate_wrong']),
        denominator_scope='All evaluated finite geometric object association edges; Unknown edges are not accepted',
        formal_false_acceptance_risk=ratio(counts['formal_wrong'],counts['formally_certified']) if not counts['formal_unscorable'] else None)


def log_risk(rows,calibration_logs=None,validation_logs=None):
    logs=sorted({x['log_token'] for x in rows})
    calibration=logs[:max(1,len(logs)//5)] if calibration_logs is None else calibration_logs
    validation=logs[len(calibration):] if validation_logs is None else validation_logs
    if set(calibration)&set(validation) or set(logs)-set(calibration)-set(validation):
        raise ValueError('Calibration/validation logs overlap or omit scored provenance')
    units=[]
    for log in validation:
        accepted=[x for x in rows if x['log_token']==log and x['formal_certified']]
        if accepted:units.append(dict(log_token=log,scenes=sorted({x['scene_id'] for x in accepted}),
            accepted=len(accepted),wrong=sum(x['score']=='wrong' for x in accepted),
            unscorable=sum(x['score']=='unscorable' for x in accepted)))
    n=len(units);errors=sum(x['wrong']>0 for x in units);unknown=sum(x['unscorable'] for x in units)
    upper=(float(beta.ppf(.95,errors+1,n-errors)) if n and errors<n else 1. if n else None) if not unknown else None
    return dict(status='BLOCKED',risk_limit=.05,confidence=.95,calibration_logs=calibration,validation_logs=validation,
        validation_log_units_with_formal_accepts=n,units=units,scene_error_events=errors if n else None,
        unscorable_formal_accepts=unknown,fully_scorable_log_units=sum(u['unscorable']==0 for u in units),
        confidence_upper_95=upper,
        reasons=['No independent qualification issued','Exploratory previously exposed population',
                 'Insufficient independently scored qualified acceptance logs'],qualification_file_generated=False)


def score_run(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/('protocol.json' if (root/'protocol.json').exists() else 'provenance.json')).read_text())
    info_path=protocol.get('infos_path',protocol.get('dataset'))
    traces=sorted((root/'scenes').glob('*/repeat_0.jsonl')) or sorted((root/'scenes').glob('*/m1_state.jsonl'))
    rows_by_scene={p.parent.name:[json.loads(l) for l in p.read_text().splitlines()] for p in traces}
    tokens={x['precheck']['frame_id'] for rows in rows_by_scene.values() for x in rows}
    with open(info_path,'rb') as f:infos=pickle.load(f)['infos']
    selected=[i for i in infos if i['token'] in tokens]
    context=log_context_from_root('data/nuscenes')
    logs=sorted({context.log_for_scene(s) for s in rows_by_scene})
    calibration=logs[:max(1,len(logs)//5)];validation=logs[len(calibration):]
    frozen=dict(scene_log_mapping={s:context.log_for_scene(s) for s in rows_by_scene},
        calibration_logs=calibration,validation_logs=validation,
        split_rule='Lexicographic independent logs; first max(1,floor(log_count/5)) calibration',
        frozen_before_label_read=True,online_GT=False,scorer_code_sha256=sha(__file__),
        gate_m=2.,ambiguity_margin_m=.25,qualification_risk_limit=.05,confidence=.95)
    (out/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    oracle=load_oracle(selected,'data/nuscenes')  # only now, after every online trace was committed
    all_scores=[];scenes=[]
    for scene,rows in rows_by_scene.items():
        identities={};population=Counter()
        for row in rows:
            p=row['precheck'];current=[a for a in row['committed_state']['anchors']
                if a['source_frame_id']==p['frame_id'] and a['anchor_type']=='object']
            predictions=[SimpleNamespace(semantic_class=a['semantic_class'],geometry_world=SimpleNamespace(center=np.asarray(a['center_world']))) for a in current]
            known,statuses=identity_population(predictions,oracle[p['frame_id']])
            identities[p['frame_id']]={current[i]['anchor_id']:ann['instance_id'] for i,ann in known.items()}
            population.update(x['status'] for x in statuses.values())
        proposals=[x for row in rows for x in row['commit']['identity_continuity_proposals']]
        scores=score_links(proposals,identities,scene,context.log_for_scene(scene));summary=summarize(scores)
        summary.update(scene_id=scene,log_token=context.log_for_scene(scene),detection_scoring_statuses=dict(population))
        (out/f'{scene}.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/f'{scene}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in scores))
        all_scores+=scores;scenes.append(summary)
    result=dict(micro=summarize(all_scores),macro={k:distribution([s[k] for s in scenes]) for k in
        ('scorable_coverage','candidate_coverage','candidate_accuracy')},scenes=scenes,log_qualification=log_risk(all_scores,calibration,validation),
        GT_used_only_after_online_execution=True,scoring_rule='Unique class-compatible sensor-supported GT centre assignment <=2m, ambiguity margin 0.25m; no dense ownership claim',
        trace_sha256={str(p):sha(p) for p in traces},
        label_table_sha256={str(p):sha(p) for p in [Path('data/nuscenes/v1.0-trainval')/n for n in
            ('sample_annotation.json','instance.json','category.json')]})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score_run(a.root,a.output)


if __name__=='__main__':main()
