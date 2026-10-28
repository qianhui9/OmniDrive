"""Summarize observed candidate/core coverage; never counts replay as KV."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
def read(name):return json.loads((root/name).read_text())
def rows(name):return [json.loads(s) for s in (root/name).read_text().splitlines() if s.strip()]
plan=read('experiment_plan.json');timeline=rows('stable_anchor_timeline.jsonl')
policy=read('semantic_qualification.json');analysis=read('stable_core_analysis.json')
groups=defaultdict(list)
previous=None
for r in timeline:
    r=dict(r,transition=previous==r['scene_id']);previous=r['scene_id']
    domain=plan['scene_datasets'][r['scene_id']]
    groups['all'].append(r)
    groups[r['split']+'/'+domain].append(r)
    groups['scene/'+r['scene_id']].append(r)
def summary(rr):
    valid=[r for r in rr if r['anchor']['stable_state_valid']]
    lives=Counter(r['anchor']['anchor_id'] for r in valid)
    checks=Counter(c['status'] for r in rr if r['anchor_hit'] for c in r['core_fact_checks_offline'])
    hits=sum(r['anchor_hit'] and r['transition'] for r in rr)
    transitions=sum(r['transition'] for r in rr)
    return dict(frames=len(rr),scenes=len({r['scene_id'] for r in rr}),
        valid_core_frames=len(valid),valid_core_frame_fraction=len(valid)/len(rr),
        anchor_hits=hits,transitions=transitions,anchor_hit_rate=hits/transitions if transitions else None,
        valid_anchor_lifetimes=list(lives.values()),mean_valid_anchor_frames=sum(lives.values())/len(lives) if lives else 0.,
        selected_core_class_distribution=dict(Counter(f[2] for r in valid for f in r['anchor']['facts'])),
        retained_fact_checks=dict(checks),
        refresh_reasons=dict(Counter(r['anchor']['update_reason'] for r in rr if not r['anchor_hit'])))
frozen=read('calibration_frozen_rules.json')
coverage=dict(policy_sha256=analysis['policy_sha256'],qualified_semantic_types=analysis['qualified_semantic_types'],
    observed_coverage={name:summary(rr) for name,rr in groups.items()},
    qualification_rules=policy['rules'],all_calibration_trials=frozen['trials'],
    target_domain_nonzero_correct_core_observed=any(r['anchor_hit'] and all(c['status']=='correct' for c in r['core_fact_checks_offline']) for r in groups['qualification_validation/val']),
    semantic_expansion_pass=False,GT_online=False,physical_certification=False,
    grace_frames=0,old_car_rule_unchanged=True,
    semantic_value='Only observed car class with associated identity continuity; no proven incremental driving-language value, geometry, motion or topology encoded.',
    new_semantics_rejection='Other object trials lack scored edges, scene support or <=5% unknown; map/region lack independently evaluable current semantic producer.',
    partial_oracle_scope='Unknown identity/class edges remain unknown; zero observed wrong edges is not a population guarantee.',
    overall_raw_candidate_validation=analysis['validation_summary'],
    historical_exposure=plan['historical_exposure'],
    scope='14 qualification logs only, calibrated 7 / validated 7; final six acceptance logs remain unopened until representation quality passes.',
    exact_KV_executed=False)
(root/'semantic_core_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
print(json.dumps({k:v for k,v in coverage['observed_coverage'].items() if not k.startswith('scene/')},indent=2))
