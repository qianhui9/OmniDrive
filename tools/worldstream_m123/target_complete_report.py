"""R0-required utilities; historical alternative runners have been retired."""

from functools import lru_cache

import json

from pathlib import Path

from .selective_answer_reuse_report import mean, grade

@lru_cache(maxsize=512)
def qa_references(token,split):
    refs={}
    for folder in ('conv','vqa'):
        p=Path('data/nuscenes')/folder/split/(token+'.json')
        if p.exists():refs.update({q['question']:q.get('answer',q.get('answers','')) for q in json.loads(p.read_text())})
    return refs


def grade_current(q,answer,token,objects,split='train'):
    from worldstream.m3.task_conditioner import question_facts
    if question_facts(q).intent!='unsupported':return grade(q,answer,token,objects)
    from .world_guided_decode_v3_report import facts
    from .world_guided_decode_v2_quality import compare_facts
    ref=qa_references(token,split).get(q)
    expected=facts(q,ref) if isinstance(ref,str) and ref.strip() else []
    checks=compare_facts(expected,facts(q,answer)) if expected else []
    return dict(known=bool(checks),score=mean([float(c['correct']) for c in checks]),severe_error=False,
        metric='limited_exact_current_QA_fact_recall',reference=dict(source='exact_current_frame_original_QA',
        reference=ref,checks=checks,reason=None if checks else 'no_reliable_reference_facts'))
