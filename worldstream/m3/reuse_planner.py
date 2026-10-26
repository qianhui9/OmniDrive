"""WSDR-KV group planning. State/region hints never certify causal KV equality."""
from dataclasses import dataclass, asdict
import hashlib
import json
import torch
from .task_conditioner import task_profile

WSDR_VERSION='wsdr_kv_group_v1'
GROUPS=('object','ego','map')


@dataclass(frozen=True)
class ReusePlan:
    task_type: str
    task_profile: str
    relevant_groups: tuple
    stable_groups: tuple
    changed_groups: tuple
    mandatory_refresh_groups: tuple
    layout_permutation: tuple
    layout_signature: str
    m1_state_valid: bool
    m2_mapping_available: bool
    fallback_reason: str
    candidate_group_order: tuple
    m2_mapping_reason: str = 'm2_region_mapping_unavailable'
    strategy_version: str = WSDR_VERSION

    def diagnostics(self):return asdict(self)


class ReusePlanner:
    @torch.no_grad()
    def plan(self,question,expanded,layout,previous=None,history=None,context=None,m2_diagnostics=None):
        task,profile,relevant=task_profile(question)
        context=context or {};history=history or {}
        state_valid=bool(history.get('state_valid'))
        reason=None
        if not state_valid:reason='m1_snapshot_unavailable_or_invalid'
        complete=layout is not None and len(expanded.visual_group_spans)==3
        if not complete:reason='carrier_group_provenance_unavailable'
        # M2 windows flow through a global EVA suffix and dense task queries.
        # No current interface proves a window -> query/token correspondence.
        if m2_diagnostics:
            bound=m2_diagnostics.get('cache_context',{})
            if (not m2_diagnostics.get('execution_completed') or any(bound.get(k)!=context.get(k)
                    for k in ('scene_id','frame_id','timestamp'))):
                state_valid=False;reason='stale_m2_execution_context'
        groups=tuple(g for g,n in layout.groups if n) if layout is not None else GROUPS
        stable=[]
        if complete and previous is not None and previous.scope==expanded.scope and previous.signature==expanded.signature:
            old={g:(a,b) for g,a,b in previous.visual_group_spans}
            comparisons=[];names=[]
            for g,a,b in expanded.visual_group_spans:
                if g not in groups or g not in old:continue
                c,d=old[g]
                if b-a!=d-c or d>previous.length or previous.embeddings.dtype!=expanded.embeddings.dtype or previous.embeddings.device!=expanded.embeddings.device:continue
                names.append(g)
                comparisons.append((expanded.embeddings[:,a:b]==previous.embeddings[:,c:d]).all()
                    & torch.isfinite(expanded.embeddings[:,a:b]).all())
            # One small metadata transfer, no clone of carrier tensors.
            if comparisons:stable=[g for g,same in zip(names,torch.stack(comparisons).cpu().tolist()) if same]
        changed=tuple(g for g in groups if g not in stable)
        mandatory=changed if state_valid else groups
        rank={g:i for i,g in enumerate(relevant)}
        candidate=tuple(sorted(GROUPS,key=lambda g:(g not in stable,rank.get(g,len(rank)),GROUPS.index(g))))
        permutation=GROUPS
        if complete and state_valid and layout.order_contract=='group_permutation_v1':
            permutation=candidate
        elif complete and reason is None:
            reason='native_group_order_unverified'
        signature=hashlib.sha256(json.dumps(dict(version=WSDR_VERSION,groups=layout.groups if layout else None,
            contract=layout.order_contract if layout else None,permutation=permutation),sort_keys=True).encode()).hexdigest()
        return ReusePlan(task,profile,relevant,tuple(stable),changed,tuple(mandatory),permutation,
            signature,state_valid,False,reason,candidate)


def organize_carriers(carriers,layout,plan):
    """Preserve every token; only a producer-declared permutation contract permits reorder."""
    if plan.layout_permutation==GROUPS:return carriers
    if layout is None or layout.order_contract!='group_permutation_v1':
        raise ValueError('Carrier producer has no group permutation contract')
    if set(plan.layout_permutation)!=set(GROUPS) or len(plan.layout_permutation)!=3:
        raise ValueError('Invalid carrier permutation')
    parts={};cursor=0
    for group,count in layout.groups:
        parts[group]=carriers[:,cursor:cursor+count];cursor+=count
    if cursor!=carriers.shape[1]:raise ValueError('Incomplete current carrier layout')
    return torch.cat([parts[g] for g in plan.layout_permutation],dim=1)
