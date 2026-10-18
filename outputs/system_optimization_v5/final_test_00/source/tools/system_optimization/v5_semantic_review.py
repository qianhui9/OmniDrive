"""Explicit question-by-question analyst text review, NOT human/visual grounding.

The reviewed questions are frozen in vqa_semantic_review_queue.json. No keyword
rule populates these judgments. Ambiguous referents remain Unknown; this cannot
authorize an online task or estimate population opportunity from biased extras.
"""
from collections import Counter
import json

from .audit import write
from .v5 import ROOT

# (review id, scene scope, unavailable meaning / concrete dependency gap)
REVIEWS = [
('b8d9a196d7ca9058','local','Sun position and time-of-day inference; no sun/lighting state'),
('280fafbef68d73e5','global','Overpass appearance and road-network interpretation; no road topology'),
('c2d8c8e5bf62e506','global','Turn feasibility requires markings, signs and routing; no legal-turn semantics'),
('c11e2407ff984a73','regional','Parking-lot description includes appearance/layout; no parking-lot state'),
('44d3fbbf3b55b31e','local','Bus description is open-ended appearance, not a direction answer'),
('f7cdd90fb460b38b','local','Truck position AND possible activity; geometry plus grounded motion/intent needed'),
('86c9554ff6b1e7ec','global','City type inferred from roads/buildings; no building or city semantics'),
('165b5337fd99ad4d','regional','Marking recognition and legal meaning; lane control points lack both'),
('9ee88b47b4dc2d2a','regional','Barrier description and lane-restriction semantics; geometry insufficient'),
('b1ddc64357cb5518','regional','Open-ended forward road layout; topology/domain completeness Unknown'),
('1090fa3dd135c779','regional','Road-type classification; no semantic road-type field'),
('52952fd5d3f67a0f','regional','Sign/signal discovery and semantics; no signs or light state'),
('dcff8142e5a058c3','regional','Traffic-flow speed aggregates multiple vehicles; temporal motion and domain needed'),
('df22c9e61462a380','ego','Ego motion plus driving-behaviour description; behaviour semantics still missing'),
('712a1149e12f96e9','regional','Traffic density and motion behind ego; complete rear domain and time evidence'),
('43ff4da0b16bf4b9','regional','Likely legal speed limit cannot be inferred from velocity predictions alone'),
('98409cb5e590a208','regional','Visible speed-limit sign OCR/legal limit absent from state'),
('c5533dfa9fa304f7','regional','Intersection type/topology absent from lane control points'),
('fa5d151caae7336e','regional','Rear traffic condition requires regional search and aggregate scene semantics'),
('14a455f9ce47b124','regional','Road condition includes surface/appearance beyond object geometry'),
('56d993820b25d6af','regional','Lot condition is an open-ended appearance/domain question'),
('35e6efa183ee33af','regional','Absence of traffic and pedestrians requires complete rear search domain'),
('a856b32b827f5a38','regional','Traffic condition ahead aggregates agents and may include signals'),
('8ebeb2bef23f3d4f','regional','Two-domain traffic comparison needs complete front and rear aggregates'),
('d8a58b64bf942cd5','regional','Building-type semantics absent from object/map state'),
('23a57b09e413e10e','global','Area-location inference is open-ended scene semantics'),
('384263901d98aa40','global','Proceeding through access control requires gate/sign/rule/current hazard state'),
('0f68f9f48be652dc','global','Safe lane change requires complete hazards, adjacent lanes, intent and gaps'),
('fa6c701b6b197be8','regional','Establishment recognition absent from state'),
('0586e428a799bda6','regional','Building/establishment search and semantics absent from state'),
('06ff2f0b8ac9c94c','local','KEEP CLEAR meaning is a road-marking/rule question, not map geometry'),
('8932bc7449e8fcdb','regional','Traffic condition requires complete road-domain traffic evidence'),
('71a18498b105158f','regional','Road surroundings need appearance/building/land-use semantics'),
('5ac7cf5545bd43c9','regional','Road condition is appearance-sensitive and open-ended'),
('573de1445e4d5fc0','regional','Forward road condition needs surface/appearance evidence'),
('1aa9a4f1cd9aee1c','regional','Building recognition absent from the current decoded fields'),
('daebac2de29a27b0','regional','Junction type requires topology/semantic map labels'),
('5e6178cf62939d8e','local','White gate identity and function require colour/appearance/access-control semantics'),
('9009c424849487e0','ego','Ego moving/stationary depends on ego velocity plus a defined decision boundary; M1 has no certified ego-motion fact'),
('4087aff339b2b7a6','ego','Ego moving/stationary depends on ego velocity plus a defined decision boundary; M1 has no certified ego-motion fact'),
('62be2813e1997db8','global','Driver responsibility is a policy/safety question requiring crossings and rules'),
('b2f4ca06878f8f37','regional','Right-rear lane traffic motion needs lane membership, multiple tracks and domain coverage'),
('f9f449c091fc8f58','local','Marking meaning absent from lane control points'),
('829c09011e5a208d','regional','Causal explanation of empty road is not a local direction fact'),
('821cae3cc7dc0ceb','regional','Road-marking recognition absent from state'),
('add07f04df05bfd1','local','Rear vehicle description needs unique visual referent and appearance attributes'),
('0ba5a01222340f16','regional','Rear road condition requires regional appearance evidence'),
('a89742247b7e80a9','regional','Right-side area description requires scene/appearance semantics'),
]


def main():
    frozen=json.loads((ROOT/'vqa_semantic_review_queue.json').read_text())
    lookup={r['review_id']:r for r in frozen['records']}
    if set(lookup)!={r[0] for r in REVIEWS}:raise ValueError('Review text does not match frozen queue')
    reviewed=[]
    for rid,scope,gap in REVIEWS:
        row=dict(lookup[rid]);row.update(review_status='analyst_text_reviewed_visual_grounding_unknown',
            annotation_kind='Explicit AI analyst text review; not human annotation or certified physical/task proof')
        row['semantic_review']=dict(unique_referent=True if scope=='ego' else 'Unknown',
            requires_full_scene=scope=='global',required_domain=scope,
            history_opportunity='conditional ego motion' if scope=='ego' else 'Unknown',
            current_head_sufficient=False,missing_semantics=gap,
            physical_error_constraint='Unknown until task and visual referent grounded',
            omittable_language_input='None verified',input_complete=False,rejection_reason=gap)
        reviewed.append(row)
    records=[json.loads(line) for line in open('outputs/system_optimization_v4/task_dependency_records.jsonl')]
    directed={
        'What is the position of the bus in relation to the vehicle?':
            ('conditional_relative_direction','A unique bus could use one signed spatial relation; question does not specify metric precision. Bus uniqueness and requested answer granularity remain Unknown without visual grounding.'),
        'What is the position of the black SUV seen from the left-rear side?':
            ('conditional_local_position','Black/SUV appearance is unavailable; class car cannot uniquely identify the referent. Position granularity is unspecified.'),
        'Is the truck directly behind us in the left, center, or right lane?':
            ('multiple_local_facts','Truck identity plus lane topology/membership required; ego lateral sign does not determine lane membership.'),
    }
    extras=[]
    for record in records:
        if record['question'] not in directed:continue
        kind,gap=directed[record['question']]
        extras.append(dict(record,review_status='analyst_text_reviewed_visual_grounding_unknown',
            annotation_kind='Purposive question-only review; not human labels, not representative coverage',
            dependency_type=kind,unique_referent='Unknown',requires_full_scene=False,
            current_head_sufficient='Unknown',input_complete=False,
            physical_error_constraint='Signed interval excludes zero for pure relative direction; task granularity Unknown',
            possible_omitted_input='Full carriers only if grounded task complete and quality/cost certified',
            rejection_reason=gap))
    write(ROOT/'vqa_semantic_dependency_reviews.json',dict(stratified=reviewed,purposive=extras))
    result=dict(original_vqa_questions=72189,lexical_candidate_count=4881,
        lexical_candidate_coverage=4881/72189,stratified_review_count=len(reviewed),
        reviewed_fraction_of_lexical_candidates=len(reviewed)/4881,
        reviewed_scenes=len({r['scene_token'] for r in reviewed}),
        review_scope='AI analyst text-only; no human or image grounding claimed',
        stratified_domains=dict(Counter(r['semantic_review']['required_domain'] for r in reviewed)),
        purposive_reviews=len(extras),conditional_direction_tasks=[r for r in extras if r['dependency_type']=='conditional_relative_direction'],
        verified_input_complete_opportunity=0,physically_certifiable_opportunity=0,
        actual_task_fact_coverage=0.,remaining_unreviewed_candidates=4881-len(reviewed)-len(extras),
        unreviewed_status='Unknown; do not infer all impossible or all usable',
        potential_token_reduction_verified=0,potential_conditional_reduction_per_task='At most 513 carriers minus replacement tokens; requires referent, semantics, physical risk and task quality gates',
        expected_verification_cost_ms=None,verification_cost_reason='No complete grounding/foreground implementation; certifier-only latency is not total verification cost',
        direct_query_approved=False,qa_quality=None,qa_quality_reason='No legal substitution; original QA not replaced')
    write(ROOT/'task_opportunity_results.json',result)
    lines=['# 原始 VQA 语义依赖复核','',
        '固定种子按城市、词法类别和场景抽取 48 条，另对 3 条明确位置问题作定向文本复核。',
        '以下是 AI 分析者逐条文本审阅，不是人工标注、图像指称确认或任务输入完备性认证。未审阅样本保留 Unknown。','',
        '| 原始问题 | 域 | 缺失依赖 |','|---|---|---|']
    lines.extend(f'| {r["question"]} | {r["semantic_review"]["required_domain"]} | {r["semantic_review"]["missing_semantics"]} |' for r in reviewed)
    lines+=['','定向检查发现真实问题 “What is the position of the bus in relation to the vehicle?”。',
        '这是有条件的单关系机会，仍未证明图像中只有一个可指称 bus，也未冻结自然语言“position”的完整答案要求。',
        '黑色 SUV 位置需要外观语义；卡车位于哪条车道需要身份、车道几何与拓扑共同支持。',
        '默认 Planning 与完整安全任务仍需要完整当前场景。当前 Head 提供预测字段，不提供完整语义、独立物理真值或查询指称。',
        '不能用 48 个未完成视觉 grounding 的文本样本推断 4,881 条全部不可替代。已验证可省略语言输入为零，条件机会与未知部分单列。']
    (ROOT/'vqa_semantic_dependency_audit.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
