"""Offline-only continuity metrics and read-only saved-trace reconstruction.

A visible GT object is an annotation with sensor support. A partial-oracle
ambiguity is a boundary, not a missing detection. GT absence is also a boundary.
A fragment requires >=1 fully visible, unambiguously unmatched frame followed
by a different predicted identity in the same visibility run. A surviving ID
bridges the observation gap without fragmenting its track. Immediate ID changes are
switches only, never fragments. No labels enter online state.
"""
import numpy as np


class ContinuityMetrics:
    def __init__(self):
        self.states={}
        self.counts=dict(predicted_track_fragments=0,continuous_identity_switches=0,
                         scored_visible=0,visible_known_missed=0,ambiguous_visible=0,bridged_visible_gaps=0)

    def update(self, annotations, predicted_by_instance, possible):
        visible={a['instance_id'] for a in annotations if a['has_sensor_support']}
        self.states={k:v for k,v in self.states.items() if k in visible}
        for identity in visible:
            aid=predicted_by_instance.get(identity)
            state=self.states.get(identity)
            if aid is None:
                if identity in possible:
                    self.counts['ambiguous_visible']+=1
                    self.states.pop(identity,None)
                else:
                    self.counts['visible_known_missed']+=1
                    if state:state.update(interrupted=True,previous_scored=False)
                continue
            self.counts['scored_visible']+=1
            if state:
                if state['interrupted']:
                    if state['aid']!=aid:self.counts['predicted_track_fragments']+=1
                    else:self.counts['bridged_visible_gaps']+=1
                elif state['previous_scored'] and state['aid']!=aid:self.counts['continuous_identity_switches']+=1
            self.states[identity]=dict(aid=aid,previous_scored=True,interrupted=False)
        return dict(self.counts)


def possible_instances(predictions,annotations):
    result=set()
    for a in annotations:
        if not a['has_sensor_support']:continue
        for p in predictions:
            if p.semantic_class!=a['class'] or p.geometry_world.size is None:continue
            ratio=np.maximum(p.geometry_world.size/a['geometry'].size,a['geometry'].size/p.geometry_world.size)
            if (ratio<=2).all() and np.linalg.norm(p.geometry_world.center-a['geometry'].center)<=1.:
                result.add(a['instance_id']);break
    return result


class SavedAssociations:
    """Reconstruct ONLY creation counters from saved pairs and fixed selection.

    Does not instantiate or run any historical method. Validate each saved
    scored relation before using reconstructed full mappings for new metrics.
    """
    def __init__(self, scene, traces):
        self.traces={t['frame_id']:t for t in traces};self.next_id=0;self.previous={}
        ids=[aid for t in traces for aid,_ in t['pairs']]
        self.prefix=ids[0].rsplit(':a',1)[0] if ids else scene+':g2'
        self.continuity=ContinuityMetrics();self.checked_edges=0

    def update(self, frame_id, selected, labels, annotations, possible):
        trace=self.traces[frame_id];pairs={int(index):aid for aid,index in trace['pairs']}
        assert all(index in selected for index in pairs), 'Saved candidate selection mismatch'
        for edge in trace['edges']:
            assert self.previous.get(edge['anchor_id'])==edge['previous_instance'], 'Saved source label mismatch'
            assert labels.get(edge['prediction_index'])==edge['current_instance'], 'Saved target label mismatch'
            self.checked_edges+=1
        mapping={}
        for index in selected:
            aid=pairs.get(index)
            if aid is None:
                self.next_id+=1;aid=self.prefix+':a'+str(self.next_id)
            else:
                assert int(aid.rsplit(':a',1)[1])<=self.next_id, 'Saved identity creation order mismatch'
            if index in labels:mapping[aid]=labels[index]
        self.previous=mapping
        self.continuity.update(annotations,{identity:aid for aid,identity in mapping.items()},possible)
        return dict(self.continuity.counts)


def strict_labels(predictions,annotations,gate=1.):
    """Offline partial oracle: unique class/size/1m edges in BOTH directions."""
    from worldstream.m1.association import legal_assignment
    costs=np.full((len(predictions),len(annotations)),np.inf)
    for i,p in enumerate(predictions):
        for j,a in enumerate(annotations):
            if p.semantic_class!=a['class'] or not a['has_sensor_support']:continue
            if p.geometry_world.size is None:continue
            ratios=np.maximum(p.geometry_world.size/a['geometry'].size,a['geometry'].size/p.geometry_world.size)
            if np.any(ratios>2):continue
            distance=np.linalg.norm(p.geometry_world.center-a['geometry'].center)
            if distance<=gate:costs[i,j]=distance
    finite=np.isfinite(costs)
    return {i:annotations[j]['instance_id'] for i,j in legal_assignment(costs)
            if finite[i].sum()==1 and finite[:,j].sum()==1}
