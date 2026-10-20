"""Fail-closed task evidence contracts, separate from relevance and M1 validity.

Only explicit observed IDs and two output schemas are understood. Historical
ID continuity/current detector agreement never constitutes independent truth.
No annotation, oracle, semantic similarity or partial-coverage count is used.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import re

GRAMMAR = re.compile(r'What is the (distance|relative position) (?:to|of) observed object ([A-Za-z0-9_.:-]+) in meters\?')
SCHEMAS={'distance':'distance_m_2dp','relative position':'ego_xyz_m_json_2dp'}


@dataclass(frozen=True)
class TaskEvidenceContract:
    task_type: str = 'unknown'
    template: str = ''
    referent_id: str = ''
    scene_id: str = ''
    frame_id: str = ''
    timestamp: float = 0.
    source_frame: str = ''
    source_version: str = ''
    current_observation_available: bool = False
    required_fields: tuple = ('center_ego_m',)
    available_fields: tuple = ()
    referent_unique: bool = False
    task_input_complete: bool = False
    state_fresh: bool = False
    contradiction_free: bool = False
    quality_certificate: str = ''
    answer_format: str = ''
    execution_signature: str = ''
    units: str = 'm'
    independent_current_identity: bool = False


@dataclass(frozen=True)
class TaskEvidenceDecision:
    contract: TaskEvidenceContract
    candidate: str | None
    evidence_complete: bool
    quality_qualified: bool
    reason: str

    def to_dict(self):return asdict(self)


def population_task(question):
    """Descriptive census only. Classification never grants execution."""
    q=str(question).lower()
    if re.search(r'planning|trajectory|driving action|should.*(do|drive)',q):return 'planning'
    if re.search(r'safe|risk|collision|conditions|scenario|environment|weather|what.*happen|panoramic|affect|influence',q):return 'global_safety'
    if re.search(r'lane|road|intersection|topology|traffic light|drivable',q):return 'map_topology'
    if re.search(r'distance|how far|meters|metres|speed',q):return 'local_metric'
    if re.search(r'car|vehicle|pedestrian|truck|bus|left|right|front|behind|object|cycl',q):return 'object_relation'
    return 'unknown'


def _qualified(cert,c):
    # The deployment supplies an independently audited artifact; a request bool
    # or the current detector's own score cannot authorize direct execution.
    if not isinstance(cert,dict):return False
    return bool(cert.get('certificate_id')==c.quality_certificate and c.quality_certificate
        and cert.get('independent_quality_pass') is True
        and cert.get('evidence_origin')=='independent_task_validation'
        and cert.get('online_ground_truth_used') is False
        and cert.get('protocol_sha256') and cert.get('evidence_sha256')
        and cert.get('heldout_samples',0)>0
        and cert.get('calibration_scenes') and cert.get('heldout_scenes')
        and not set(cert['calibration_scenes'])&set(cert['heldout_scenes'])
        and cert.get('execution_signature')==c.execution_signature
        and cert.get('template')==c.template and cert.get('answer_format')==c.answer_format
        and cert.get('units')==c.units and cert.get('identity_evidence_independent') is True)


def evaluate_task_evidence(question,context,records=(),certificate=None):
    match=GRAMMAR.fullmatch(str(question).strip())
    identity=dict(scene_id=str(context.get('scene_id','')),frame_id=str(context.get('frame_id','')),
        timestamp=context.get('timestamp',0.),execution_signature=str(context.get('state_execution_signature') or ''))
    c=TaskEvidenceContract(**identity,task_type=population_task(question))
    def reject(reason,contract=c,candidate=None,complete=False):
        return TaskEvidenceDecision(contract,candidate,complete,False,reason)
    if not match:return reject('unsupported_explicit_task_grammar')
    records=[r for r in (records or ()) if isinstance(r,dict)]
    metric,referent=match.groups();template=metric.replace(' ','_')+'_observed_id'
    selected=[r for r in records if r.get('referent_id',r.get('anchor_id'))==referent]
    base=dict(identity,task_type='local_metric',template=template,referent_id=referent,
        answer_format=SCHEMAS[metric],referent_unique=len(selected)==1)
    if len(selected)!=1:return reject('referent_missing_or_ambiguous',TaskEvidenceContract(**base))
    r=selected[0];fields=r.get('current_fields',{})
    if not isinstance(fields,dict):fields={}
    try:valid_time=math.isfinite(float(identity['timestamp']))
    except (TypeError,ValueError):valid_time=False
    fresh=bool(identity['scene_id'] and identity['frame_id'] and identity['execution_signature']
        and valid_time
        and (r.get('scene_id'),r.get('frame_id'),r.get('timestamp'),r.get('execution_signature'))==
        (identity['scene_id'],identity['frame_id'],identity['timestamp'],identity['execution_signature'])
        and r.get('source_frame')==identity['frame_id'] and bool(r.get('source_version'))
        and r.get('source_version')==context.get('source_version'))
    xyz=fields.get('center_ego_m');valid=isinstance(xyz,(list,tuple)) and len(xyz)==3
    valid=valid and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in xyz)
    c=TaskEvidenceContract(**base,source_frame=str(r.get('source_frame') or ''),source_version=str(r.get('source_version') or ''),
        current_observation_available=r.get('current_observation_available') is True,
        available_fields=tuple(sorted(fields)),task_input_complete=bool(valid),state_fresh=fresh,
        contradiction_free=r.get('contradiction_free') is True,
        quality_certificate=str(r.get('quality_certificate') or ''),units=str(r.get('units') or ''),
        independent_current_identity=r.get('identity_evidence_origin')=='independent_current_observation')
    for ok,why in [(c.state_fresh,'stale_source_or_signature'),(c.current_observation_available,'no_current_observation'),
            (c.independent_current_identity,'no_independent_referent_identity'),
            (r.get('evidence_origin') not in ('annotation','ground_truth','oracle'),'online_ground_truth_forbidden'),(c.task_input_complete,'missing_or_nonfinite_fields'),
            (c.contradiction_free,'current_history_contradiction'),(c.units=='m','unsupported_units')]:
        if not ok:return reject(why,c)
    import json
    candidate=f'{math.hypot(xyz[0],xyz[1]):.2f} m' if metric=='distance' else json.dumps(dict(x_m=round(xyz[0],2),y_m=round(xyz[1],2),z_m=round(xyz[2],2)),separators=(',',':'),allow_nan=False)
    qualified=_qualified(certificate,c)
    return TaskEvidenceDecision(c,candidate,True,qualified,'independently_qualified_task' if qualified else 'missing_independent_task_quality_certificate')


def native_prompt_matches(tokenizer,input_ids,question):
    """Bind metadata to the one native USER turn before any actual shortcut.

    Unknown prompt wrappers/history/system instructions fail closed. The only
    supported schema is the original OmniDrive Vicuna single-turn template.
    """
    system=("A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.")
    try:
        flat=input_ids.reshape(-1)
        if int((flat==-200).sum())!=1:return False
        text=tokenizer.decode(flat[flat>=0].tolist(),skip_special_tokens=True).strip()
    except (AttributeError,TypeError,ValueError):return False
    pattern=re.escape(system)+r' USER:\s*You are driving in (?:boston|singapore)\. '+re.escape(question)+r' ASSISTANT:'
    return re.fullmatch(pattern,text) is not None
