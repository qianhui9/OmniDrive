"""Offline-only policies. No production configuration enables this adapter.

Fixed research profile (not calibrated): score >= .5, three consecutive
committed alignments, source-bound freshness <= 1 second. Scores never rank
association edges. External metadata is published only on the transaction fork.
"""
from dataclasses import replace
import copy
import numpy as np
from worldstream.m1 import WorldStateManager
from worldstream.m1.association import associate
from worldstream.m1.dqtrack_bridge import align


def key(h):
    return tuple(h['namespace']) + (h['tracking_id'],)


def fresh(anchor, frame, epoch, origin):
    h=anchor.external_tracking
    return bool(h and h.get('count',0)>=3 and h.get('tracking_score',0)>=.5
        and h.get('namespace',[None,None])[1]==frame.scene_id
        and h.get('frame_id')==anchor.source_frame_id
        and h.get('timestamp')==anchor.source_timestamp
        and h.get('state_version')==anchor.state_version
        and h.get('epoch')==epoch==anchor.source_state_epoch
        and h.get('signature')==frame.execution_signature==anchor.source_execution_signature
        and h.get('origin')==origin==anchor.evidence_origin
        and 0<frame.timestamp-h['timestamp']<=1.)


def choose(anchors,predictions,pairs,costs,hints,frame,epoch,origin,method):
    """B only vetoes. C replaces a whole complete legal competing component."""
    hints={j:h for j,h in hints.items() if h.get('status')=='aligned'
        and h.get('frame_id')==frame.frame_id and h.get('timestamp')==frame.timestamp
        and h.get('namespace',[None,None])[1]==frame.scene_id and h.get('tracking_score',0)>=.5}
    # Duplicate identities can never prove an external one-to-one mapping.
    keys=[key(h) for h in hints.values()]
    hints={j:h for j,h in hints.items() if keys.count(key(h))==1}
    old={i:a.external_tracking for i,a in enumerate(anchors) if a.anchor_type=='object'
         and fresh(a,frame,epoch,origin)}
    old_keys=[key(h) for h in old.values()]
    old={i:h for i,h in old.items() if old_keys.count(key(h))==1}
    result=[];vetoes=[];resolved=[];components=[]
    for i,j in pairs:
        if (i in old and j in hints and old[i]['namespace']==hints[j]['namespace']
                and key(old[i])!=key(hints[j]) and not anchors[i].identity_link_qualified):
            vetoes.append([i,j])
        else:result.append((i,j))
    if method=='C':
        finite=np.isfinite(costs);seen=set()
        for first in range(len(anchors)):
            if first in seen or not finite[first].any():continue
            rows={first};cols=set();pending=[first]
            while pending:
                i=pending.pop();seen.add(i)
                for j in np.flatnonzero(finite[i]):
                    if int(j) in cols:continue
                    cols.add(int(j))
                    for r in np.flatnonzero(finite[:,j]):
                        if int(r) not in rows:rows.add(int(r));pending.append(int(r))
            if len(rows)==len(cols)==1:continue
            reason='incomplete_or_nonunique_external_mapping';mapping=[]
            if len(rows)==len(cols) and rows<=old.keys() and cols<=hints.keys():
                by_key={key(hints[j]):j for j in cols}
                mapping=[(i,by_key[key(old[i])]) for i in sorted(rows) if key(old[i]) in by_key]
                if (len(mapping)==len(rows) and all(finite[i,j] for i,j in mapping)
                        and not any(anchors[i].identity_link_qualified for i in rows)):
                    result=[(i,j) for i,j in result if i not in rows and j not in cols]+mapping
                    resolved.extend([list(p) for p in mapping]);reason='resolved_complete_component'
            components.append(dict(rows=sorted(rows),cols=sorted(cols),reason=reason))
    assert len({i for i,j in result})==len(result)==len({j for i,j in result})
    assert all(np.isfinite(costs[i,j]) for i,j in result)
    return result,dict(vetoes=vetoes,resolved=resolved,components=components)


class OfflineDQManager(WorldStateManager):
    """Uses normal precheck/fork/commit/rollback; never enabled by M1Config."""
    def __init__(self,config,method,path):
        if method not in ('B','C') or config.dqtrack_mode!='off':raise ValueError('Offline B/C with off config required')
        super().__init__(config);self.research_method=method;self.research_path=str(path)

    def _associate(self,anchors,predictions,timestamp,config,*args,**kwargs):
        pairs,costs=associate(anchors,predictions,timestamp,config,*args,**kwargs)
        frame,epoch,origin=kwargs['source_context']
        hints,diagnostics=align(frame,predictions,replace(config,dqtrack_results_path=self.research_path))
        selected,policy=choose(anchors,predictions,pairs,costs,hints,frame,epoch,origin,self.research_method)
        self._research_hints=hints
        self._research_prior={a.anchor_id:copy.deepcopy(a.external_tracking) for a in anchors}
        diagnostics['policy']=policy;diagnostics['baseline_pairs']=[[anchors[i].anchor_id,j] for i,j in pairs]
        self._research_diagnostics=diagnostics
        return selected,costs

    def _commit(self,frame,outputs,result):
        value=super()._commit(frame,outputs,result)
        if self.last_commit_diagnostics.get('input_rejection'):return value
        # This method runs on WorldStateManager.commit's private worker/fork.
        for j,aid in enumerate(self.last_commit_diagnostics['committed_anchor_ids']):
            a=self.store.anchors.get(aid)
            if a is None:continue
            h=self._research_hints.get(j);previous=self._research_prior.get(aid,{})
            if h is None:a.external_tracking={};continue
            consecutive=(previous and previous.get('tracking_score',0)>=.5 and h.get('tracking_score',0)>=.5 and key(previous)==key(h)
                and 0<frame.timestamp-previous['timestamp']<=1.
                and previous.get('signature')==frame.execution_signature
                and previous.get('epoch')==self.store.generation
                and previous.get('origin')==outputs.source)
            a.external_tracking=dict(copy.deepcopy(h),count=(previous.get('count',0)+1 if consecutive else 1) if h.get('tracking_score',0)>=.5 else 0,
                epoch=self.store.generation,signature=frame.execution_signature,origin=outputs.source,state_version=a.state_version)
        self.last_commit_diagnostics['dqtrack']=self._research_diagnostics
        return value
