"""Stratified original-question review queue; never turns lexical bins into truth."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import random

from .audit import write
from .v5 import ROOT

FIELDS = ('unique_referent','requires_full_scene','history_opportunity','current_head_sufficient',
          'missing_semantics','physical_error_constraint','omittable_language_input',
          'input_complete','rejection_reason')


def main():
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    city={i['scene_token']:('boston' if i['location'].startswith('boston') else 'singapore') for i in infos}
    source=Path('outputs/system_optimization_v4/task_dependency_records.jsonl')
    records=[json.loads(line) for line in source.read_text().splitlines()]
    candidates=[r for r in records if r['source'] in ('conv','vqa') and r['semantic_candidate']]
    groups=defaultdict(lambda:defaultdict(list))
    for row in candidates: groups[(city[row['scene_token']],row['category'])][row['scene_token']].append(row)
    rng=random.Random(20261018);sample=[]
    for key,scenes in sorted(groups.items()):
        ordered=sorted(scenes);rng.shuffle(ordered)
        # Six scene-stratified examples in each city/category, never answer-selected.
        for scene in ordered[:6]:
            row=dict(rng.choice(scenes[scene]));row['city']=key[0]
            row['review_id']=hashlib.sha256(json.dumps([row['sample_token'],row['source'],row['question']]).encode()).hexdigest()[:16]
            row['original_task_category']=row['category']
            row['review_status']='pending_review'
            row['annotation_kind']='review_queue; not human annotation'
            row['current_state_fields']=['class','3d_center','box_size','unverified_velocity','lane_control_points']
            row['semantic_review']={field:'Unknown' for field in FIELDS}
            sample.append(row)
    path=ROOT/'vqa_semantic_review_queue.json'
    if path.exists():raise FileExistsError('Review selection frozen; do not resample')
    write(path,dict(seed=20261018,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        lexical_candidates=len(candidates),sample_count=len(sample),
        sampling='6 distinct scenes per city/lexical category; question-only fixed-seed sampling',records=sample))
    print(json.dumps([dict(id=r['review_id'],scene=r['scene_token'],city=r['city'],category=r['category'],question=r['question']) for r in sample],indent=2))


if __name__=='__main__':main()
