"""Offline quality scoring after the long-history state requests have finished."""
import json,pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from .audit import write,stats
from .identity import token_index


def main(root,subdirectory='long_state_supply'):
    from tools.worldstream_m3.experiment import ground_truth
    from tools.worldstream_m3.offline_reference import resolve_reference
    from tools.worldstream_m3.llm_probe import quality
    from tools.worldstream_m3.qa_analysis import planning_quality
    root=Path(root);path=root/subdirectory/'queries.jsonl'
    if not path.exists():return None
    infos=token_index(pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos'])
    records=[]
    for line in path.open():
        row=json.loads(line);tasks=[]
        for question,native,candidate,equal in zip(row['questions'],row['native_text'],row['candidate_text'],row['complete_question_token_equality']):
            gt,origin=resolve_reference(row['token'],question,ground_truth(row['token']))
            planning='planning trajectory' in question.lower()
            tasks.append(dict(question=question,native_text=native,candidate_text=candidate,complete_token_equality=equal,
                ground_truth=gt,ground_truth_source=origin,
                native_qa=quality(native,gt) if gt is not None else None,candidate_qa=quality(candidate,gt) if gt is not None else None,
                native_planning=planning_quality(native,infos[row['token']]) if planning else None,
                candidate_planning=planning_quality(candidate,infos[row['token']]) if planning else None))
        records.append(dict(mode=row['mode'],scene=row['scene'],token=row['token'],ordinal=row['ordinal'],tasks=tasks))
    modes=[]
    for mode in sorted({r['mode'] for r in records}):
        tasks=[t for r in records if r['mode']==mode for t in r['tasks']]
        row=dict(mode=mode,questions=len(tasks),complete_tokens_equal=sum(t['complete_token_equality'] for t in tasks),
            qa_scorable=sum(t['native_qa'] is not None for t in tasks))
        for backend in ('native','candidate'):
            scored=[t[backend+'_qa'] for t in tasks if t[backend+'_qa'] is not None]
            planning=[t[backend+'_planning'] for t in tasks if t[backend+'_planning'] is not None]
            row[backend]=dict(mean_lexical_token_f1=float(np.mean([s['token_f1'] for s in scored])) if scored else None,
                planning_parsed=sum(p.get('status')=='parsed' for p in planning),planning_questions=len(planning),
                ade_m=stats([p['ade_m'] for p in planning if p.get('ade_m') is not None]),
                fde_m=stats([p['fde_m'] for p in planning if p.get('fde_m') is not None]))
        modes.append(row)
    result=dict(scope='Independent annotation scoring after all model generations; no online GT',modes=modes,records=records,
        caveat='Annotation text overlap and trajectory distance are reported separately; no LLM judge or closed-loop safety claim')
    write(root/subdirectory/'task_quality.json',result);return result
