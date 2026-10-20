"""Cross-check frozen physical history with current predictions, after heads.

Agreement is prediction consistency, not independent physical or task truth.
This API never changes M1/M2 decisions or writes current outputs into history.
"""
from dataclasses import replace
import json,re
import numpy as np


def post_verify(state,frame,outputs,gate_m=.5):
    from .world_state import VerifiedWorldState
    if state is None:return None
    identity=(frame.scene_id,frame.frame_id,frame.timestamp,frame.execution_signature)
    if identity!=(state.scene_id,state.frame_id,state.timestamp,state.execution_signature):
        return replace(state,records=(),diagnostics=dict(reason='stale_post_perception',task_quality_certified=False))
    predictions=[p for p in outputs.anchors if p.anchor_type=='object' and p.source_score is not None and p.source_score>=.3]
    history=[r for r in state.records if r['anchor_type']=='object']
    records=[];matches=[]
    pose=frame.ego_to_world
    for record in history:
        if record.get('source_frame')==frame.frame_id or record['source_timestamp']>=frame.timestamp:continue
        center=pose[:3,:3]@np.asarray(record['center_ego_m'])+pose[:3,3]
        eligible=[(j,p) for j,p in enumerate(predictions) if p.semantic_class==record['semantic_class']]
        distances=sorted((float(np.linalg.norm(p.geometry_world.center-center)),j) for j,p in eligible)
        if not distances or distances[0][0]>gate_m:continue
        distance,j=distances[0]
        if len(distances)>1 and distances[1][0]-distance<gate_m:continue
        # Mutual uniqueness: one prediction cannot certify multiple histories.
        competitors=[r for r in history if r['semantic_class']==record['semantic_class']]
        reverse=[(float(np.linalg.norm(pose[:3,:3]@np.asarray(r['center_ego_m'])+pose[:3,3]-predictions[j].geometry_world.center)),r['anchor_id']) for r in competitors]
        reverse.sort()
        if reverse[0][1]!=record['anchor_id'] or (len(reverse)>1 and reverse[1][0]-reverse[0][0]<gate_m):continue
        provenance=dict(stage='post_perception',current_frame=frame.frame_id,current_timestamp=frame.timestamp,
            current_execution_signature=frame.execution_signature,current_prediction=j,center_residual_m=distance,
            physical_certificate_inherited=True,independent_physical_truth=False,task_quality_certified=False)
        records.append(dict(record,post_perception=provenance));matches.append(provenance)
    return VerifiedWorldState(state.scene_id,state.frame_id,state.timestamp,state.state_epoch,state.execution_signature,
        tuple(records),dict(state.diagnostics,stage='post_perception',historical_candidate_count=state.diagnostics.get('total_history_anchors',len(state.records)),
            physical_verified_count=len(state.records),task_verified_candidates=len(records),prediction_matches=matches,
            current_head_is_prediction=True,task_quality_certified=False,complete_scene_coverage=False))


def local_query(state,question,context):
    """Only narrowly specified local metrics; no counting/nearest/absence claims."""
    reject=lambda why:dict(admitted=False,reason=why,records=[],text='',task='local_metric')
    if state is None or state.diagnostics.get('stage')!='post_perception':return reject('no_post_verified_state')
    if (state.scene_id,state.frame_id,state.timestamp,state.state_epoch,state.execution_signature)!=(
            str(context.get('scene_id')),str(context.get('frame_id')),context.get('timestamp'),context.get('state_epoch'),context.get('state_execution_signature')):
        return reject('stale_post_state')
    q=question.strip().lower()
    # Explicit grammar avoids interpreting unrestricted QA as an input-complete task.
    match=re.fullmatch(r'what is the (distance|relative position) (?:to|of) the (car|truck|bus|pedestrian|traffic cone|barrier) (in front|behind|on the left|on the right)(?: of (?:us|the ego vehicle))?\?',q)
    if not match:return reject('original_question_requires_full_visual_context')
    metric,cls,side=match.groups();cls=cls.replace(' ','_')
    selected=[r for r in state.records if r['semantic_class']==cls]
    axis,sign=(0,1) if side=='in front' else (0,-1) if side=='behind' else (1,1) if side=='on the left' else (1,-1)
    selected=[r for r in selected if r['center_ego_m'][axis]*sign>0]
    if len(selected)!=1:return reject('target_identity_ambiguous_or_unknown')
    record=selected[0]
    # Do not infer that this is the only object in the scene, only one observed referent.
    payload=dict(observed_referent=record,complete_scene_coverage=False,metric=metric)
    return dict(admitted=True,reason='offline_post_verified_local_candidate',records=selected,task='local_metric',
        text='\nPartial historical observation cross-checked with current predictions; not ground truth:\n'+json.dumps(payload,separators=(',',':'))+'\n',
        task_quality_certified=False,complete_scene_coverage=False)


def direct_query(state,question,context,task_certificate=None):
    """Compatibility entry shares the strict serving gate, not head agreement."""
    from .task_evidence import evaluate_task_evidence
    result=evaluate_task_evidence(question,context,getattr(state,'records',()),task_certificate)
    return dict(result.to_dict(),admitted=result.evidence_complete and result.quality_qualified,
        text=result.candidate if result.quality_qualified else '',task=result.contract.task_type)
