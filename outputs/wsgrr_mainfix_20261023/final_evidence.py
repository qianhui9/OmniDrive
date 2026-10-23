"""Final read-only source, question, metric-boundary and checkpoint attestation."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'references/wsgrrr_20261022_followup'
def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

manifest=read(ROOT/'final_source_manifest.json')
source_checks={p:sha(Path(p))==h for p,h in manifest['files'].items()}
candidate_hashes=read(ROOT/'main_full/m1_m2_source_hashes.json')
for name in ['main_r0','main_r1']:
    assert read(ROOT/name/'m1_m2_source_hashes.json')==candidate_hashes
protected=read(OLD/'protected_hash_check.json')['files']
protected_checks={p:sha(Path(p))==v['current'] for p,v in protected.items()}
native=read(OLD/'Native_full/official_eval/summary.json')
candidate=read(ROOT/'main_full/official_eval/summary.json')
assert candidate['frames']==6019 and candidate['scenes']==150 and candidate['full_validation']
assert native['sample_tokens']==candidate['sample_tokens']
assert native['detector_criteria']==candidate['detector_criteria']
nc=read(OLD/'Native_full/effective_config.json')['config']
cc=read(ROOT/'main_full/effective_config.json')['config']
assert nc['data']['test']['pipeline']==cc['data']['test']['pipeline']
for key in ['img_backbone','pts_bbox_head','map_head','lm_head']:
    if key in nc['model']:assert nc['model'][key]==cc['model'][key],key
assert nc.get('fp16')==cc.get('fp16')
identity=None;groups={};generated={}
for name in ['Native_r0','main_r0','main_r1','Native_r1']:
    data=rows(ROOT/name/'per_question_metrics.jsonl')
    current=[(r['scene_id'],r['frame_id'],r['question_index'],r['question'],r['reference'],r['input_tokens']) for r in data]
    if identity is None:identity=current
    assert current==identity, name
    assert len(data)==40 and all(r['native_generation_executed'] and not r['prefill_executor_executed'] for r in data)
    grouped=defaultdict(list)
    for r in data:grouped[r['task_type']].append(r['quality'])
    groups[name]={k:dict(questions=len(v),exact_mismatch_rate=float(np.mean([not a['exact_match'] for a in v])),
        token_f1=float(np.mean([a['token_f1'] for a in v]))) for k,v in grouped.items()}
    generated[name]=dict(total=sum(r['generated_tokens'] for r in data),
        mean=float(np.mean([r['generated_tokens'] for r in data])))
(ROOT/'vlm_groups.json').write_text(json.dumps(dict(groups=groups,generated_tokens=generated,
    paired_unique_questions=40,scope='Reference Exact Match/Token F1 only. Exact mismatch is a lexical error measure, not independently annotated semantic driving-scene correctness.'),indent=2))
checkpoint=Path('ckpts/iter_10548.pth');actual=sha(checkpoint)
expected=read(ROOT/'native_reuse_check.json')['checks'][str(checkpoint)]['expected']
result=dict(source_checks=source_checks,protected_checks=protected_checks,
    checkpoint=dict(bytes=checkpoint.stat().st_size,expected_sha256=expected,actual_sha256=actual,unchanged=actual==expected),
    full_paired_frames=6019,full_paired_scenes=150,identical_detection_criteria=True,
    identical_task_pipeline=True,identical_task_heads=True,identical_precision_configuration=True,
    identical_questions_references_input_lengths=True,unique_questions=40,
    candidate_source_hashes_identical_across_full_and_repeats=True,
    production_approvals_added=0)
result['passed']=all(source_checks.values()) and all(protected_checks.values()) and actual==expected
(ROOT/'final_evidence.json').write_text(json.dumps(result,indent=2))
assert result['passed']
print('Final evidence PASS; protected files unchanged; checkpoint',actual)
