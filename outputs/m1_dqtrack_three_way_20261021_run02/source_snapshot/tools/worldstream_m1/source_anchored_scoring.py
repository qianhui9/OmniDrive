"""Additional OFFLINE diagnostic; does not replace the original target scorer.

Choose sparse two-frame reference by SOURCE pixel, so a wrong target cannot
select a convenient reference. Same 1px sparse radius and 2px error threshold;
local displacement approximation and sparse sensor/rigid-motion limitations
remain. All absent/ambiguous references stay Unknown, on a shared source set.
"""
import numpy as np
from scipy.spatial import cKDTree


def score_source_anchored(reference,measurements):
    p=np.asarray(measurements['source_points'],float)
    q=np.asarray(measurements['current_points'],float)
    ids=np.flatnonzero(reference['scorable'])
    rows=[dict(index=j,score='unscorable',correspondence_error_px=None,
               reason='no_sparse_source_reference') for j in range(len(p))]
    if not len(ids):return rows
    tree=cKDTree(reference['source_uv'][ids]);distance,index=tree.query(p)
    for j in np.flatnonzero(distance<=1.):
        i=ids[index[j]];near=ids[tree.query_ball_point(p[j],1.)]
        predictions=reference['target_uv'][near]+p[j]-reference['source_uv'][near]
        if (len(predictions)>1 and np.linalg.norm(predictions-predictions[0],axis=1).max()>2.):
            rows[j]['reason']='ambiguous_sparse_source_surfaces';continue
        target=reference['target_uv'][i]+p[j]-reference['source_uv'][i]
        error=float(np.linalg.norm(q[j]-target))
        if not np.isfinite(error):continue
        rows[j].update(score='correct' if error<=2. else 'wrong',correspondence_error_px=error,
                       reason='both_frames_sparse_source_anchored')
    return rows
