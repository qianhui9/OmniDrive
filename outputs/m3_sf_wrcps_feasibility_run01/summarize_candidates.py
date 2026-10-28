"""Separate candidate, qualified-fact and one-core counts from the raw audit."""
from collections import Counter,defaultdict
import json
from pathlib import Path
root=Path(__file__).resolve().parent
totals=defaultdict(Counter)
with (root/'core_candidate_timeline.jsonl').open('w') as out:
    for line in (root/'state_audit/frames.jsonl').open():
        r=json.loads(line)
        counts=dict(object_candidates=len(r['candidates']['records']),
            eligible_observations=sum(bool(a['eligible_observation']) for a in r['candidates']['records']),
            qualified_facts=len(r['selection']['facts']),selected_core_facts=len(r['anchor']['facts']),
            valid_core_frames=int(r['anchor']['stable_state_valid']),anchor_extensions=int(r['anchor_hit']))
        assert counts['selected_core_facts']<=1 and counts['qualified_facts']>=counts['selected_core_facts']
        for key in ('all',r['split'],'scene/'+r['scene_id']):totals[key].update(dict(frames=1,**counts))
        out.write(json.dumps(dict(scene_id=r['scene_id'],frame_id=r['frame_id'],split=r['split'],**counts))+'\n')
result=dict(scope='Full raw32 candidate-pool audit; fact observations can exceed one committed core; not training-enriched/deployment coverage.',
    groups={k:dict(v) for k,v in totals.items()})
(root/'candidate_fact_distribution.json').write_text(json.dumps(result,indent=2)+'\n')
path=root/'data_feasibility_report.md';text=path.read_text().split('\n## 候选与单核心分母补充')[0]
summary={k:dict(totals[k]) for k in ('all','train','validation')}
text+='\n## 候选与单核心分母补充\n\n|范围|帧|候选|连续证据eligible|合格事实观察|选入核心事实|延续|\n|---|---:|---:|---:|---:|---:|---:|\n'
for key,values in summary.items():
    text+='|'+key+'|'+ '|'.join(str(values[n]) for n in ('frames','object_candidates','eligible_observations','qualified_facts','selected_core_facts','anchor_extensions'))+'|\n'
text+='\n此处可评分观察指在线证据eligible，不代表离线GT已评分；候选与合格事实观察数可超过每帧一个核心。逐帧数字见core_candidate_timeline.jsonl，逐场景见candidate_fact_distribution.json。\n'
path.write_text(text)
print(json.dumps(summary,indent=2))
