"""Analyze saved results only; no association/model/baseline execution."""
import ast
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
MAIN=ROOT/'main_method'
s=json.loads((MAIN/'summary.json').read_text())
legacy=json.loads(Path('outputs/m1_dqtrack_policy_20261021/analysis_summary_owned.json').read_text())['methods']['geometry']['metrics']
refined=json.loads(Path('outputs/m1_world_identity_refinement_20261021_run01/main_method/summary.json').read_text())['metrics']
new_correct=set();old_correct=set();new_wrong=set();old_wrong=set();times=[];stats=[];opportunities=[];raw_recovery=[];by_scene={};saved_cont={}
for scene in s['scenes']:
    n=json.loads((MAIN/(scene+'.json')).read_text())
    o=json.loads((Path('outputs/m1_dqtrack_policy_20261021/paired_owned')/scene/'geometry.json').read_text())
    assert all(n['metrics'][k]==o['metrics'][k] for k in ('candidates','eligible'))
    def correct(x):return {(t['frame_id'],e['current_instance']) for t in x['traces'] for e in t['edges'] if e['status']=='correct'}
    def wrong(x):return {(t['frame_id'],e['previous_instance'],e['current_instance']) for t in x['traces'] for e in t['edges'] if e['status']=='wrong'}
    new_correct|=correct(n);old_correct|=correct(o);new_wrong|=wrong(n);old_wrong|=wrong(o)
    times += [t['m1_ms'] for t in n['traces']]
    for t in n['traces']:
        assert t['dqtrack'] is None and not t['certified']
        d=t['association_tracks'];stats.append(d)
        assert d['tracks']<=256 and max(d['serialized_bytes'],d['accounted_memory_bytes'])<=4194304
    opportunities+=n['recovery_opportunities'];raw_recovery+=n['recoveries']
    saved_cont[scene]=n['saved_continuity_metrics']
    by_scene[scene]=dict(metrics=n['metrics'],m1=n['m1'],fast=n['strata'].get('motion:fast'))
exact=json.loads((ROOT/'exact_source_recovery.json').read_text())
diff=dict(lost_correct=[list(x) for x in sorted(old_correct-new_correct)],gained_correct=[list(x) for x in sorted(new_correct-old_correct)],
          new_immediate_wrong=[list(x) for x in sorted(new_wrong-old_wrong)])
frozen=json.loads((ROOT/'frozen_manifest.json').read_text());changed={};patch=[];delivery={}
for name,sha in frozen.items():
    current=Path(name).read_bytes();actual=hashlib.sha256(current).hexdigest();delivery[name]=actual
    archived=(ROOT/'frozen_source'/name).read_bytes();assert hashlib.sha256(archived).hexdigest()==sha
    if actual!=sha:
        changed[name]=dict(measured_sha256=sha,delivery_sha256=actual)
        patch.extend(difflib.unified_diff(archived.decode().splitlines(True),current.decode().splitlines(True),fromfile='measured/'+name,tofile='delivery/'+name))
assert set(changed)=={'worldstream/m1/association_tracks.py','worldstream/m1/config.py','tools/worldstream_m1/world_identity_eval.py'},changed
(ROOT/'post_replay_changes.patch').write_text(''.join(patch))
(ROOT/'delivery_manifest.json').write_text(json.dumps(delivery,indent=2))
(ROOT/'source_revision_note.json').write_text(json.dumps(dict(changed_after_only_replay=changed,
    reason='Unsafe age-independent association prediction budget corrected globally; supplemental recovery score now uses the latest actual source label.',
    real_data_replay_of_delivery=False,measured_numbers_apply_to='frozen_source only',second_replay_performed=False),indent=2))
# Preserve the single-allocation invariant in the final delivery.
tree=ast.parse(Path('worldstream/m1/state_manager.py').read_text())
commit=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_commit')
alloc=[n for n in ast.walk(commit) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='associate']
writes=[n for n in ast.walk(commit) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='accepted_pairs' for t in n.targets)]
assert len(alloc)==len(writes)==1
log_sizes=[len(json.dumps(d).encode()) for d in stats]
analysis=dict(decision='NO_GO_REAL_DATA_QUALITY_NOT_VALIDATED_AFTER_FIX',single_path_structure_verified=True,main_method_runs=1,
    measured_metrics=s['metrics'],saved_original=legacy,saved_refinement=refined,scenes=by_scene,
    same_immediate_wrong_relations=new_wrong==old_wrong,relation_differences=diff,
    recovery_exact_source_counts=exact['counts'],raw_last_scored_label_counts=dict(Counter(x['status'] for x in raw_recovery)),
    recovery_opportunities=len(opportunities),opportunity_types=dict(Counter(x['kind'] for x in opportunities)),
    opportunity_applied=sum(x['dormant_applied'] for x in opportunities),opportunity_identity_preserved=sum(x['identity_preserved'] for x in opportunities),
    saved_new_metric_totals={name:{k:sum(v[name][k] for v in saved_cont.values()) for k in next(iter(saved_cont.values()))[name]} for name in ('geometry','prior')},
    measured_resources=dict(frames=len(times),m1_p50_ms=float(np.median(times)),m1_p95_ms=float(np.percentile(times,95)),
        max_tracks=max(d['tracks'] for d in stats),max_history_memory_bytes=max(d['accounted_memory_bytes'] for d in stats),
        max_serialized_history_bytes=max(d['serialized_bytes'] for d in stats),max_association_image_history_bytes=max(d['image_history_bytes'] for d in stats),
        diagnostic_p50_bytes=float(np.median(log_sizes)),diagnostic_p95_bytes=float(np.percentile(log_sizes,95)),diagnostic_max_bytes=max(log_sizes)))
(ROOT/'analysis.json').write_text(json.dumps(analysis,indent=2))
(ROOT/'relation_differences.json').write_text(json.dumps(diff,indent=2))
(ROOT/'recovery_opportunities.json').write_text(json.dumps(opportunities,indent=2))
print(json.dumps({k:v for k,v in analysis.items() if k not in ('scenes','relation_differences','saved_original','saved_refinement')},indent=2))
