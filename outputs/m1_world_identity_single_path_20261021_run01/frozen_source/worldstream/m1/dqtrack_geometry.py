"""Offline diagnostic box geometry; not an online association gate."""
import numpy as np

def geometry_features(old,new):
    """Optional diagnostic oriented 3D IoU of convex box corners; no gate changes.

    Intersection is the halfspace intersection of the two actual world boxes.
    The Chebyshev center supplies a strictly interior point. Degenerate or
    numerically unresolved geometry returns unknown, never a favorable score.
    """
    from scipy.spatial import ConvexHull,HalfspaceIntersection,QhullError
    from scipy.optimize import linprog
    result=dict(center_distance_m=float(np.linalg.norm(old.center-new.center)),oriented_iou_3d=None)
    try:
        if old.points.shape!=(8,3) or new.points.shape!=(8,3):return result
        a=ConvexHull(old.points);b=ConvexHull(new.points)
        planes=np.unique(np.round(np.r_[a.equations,b.equations],12),axis=0)
        norms=np.linalg.norm(planes[:,:3],axis=1)
        lp=linprog([0,0,0,-1],A_ub=np.c_[planes[:,:3],norms],b_ub=-planes[:,3],
                   bounds=[(None,None)]*3+[(0,None)],method='highs')
        if not lp.success:
            if lp.status==2:result['oriented_iou_3d']=0.
            return result
        if lp.x[3]<=1e-9:
            result['oriented_iou_3d']=0.;return result
        vertices=HalfspaceIntersection(planes,lp.x[:3]).intersections
        volume=ConvexHull(vertices).volume
        result['oriented_iou_3d']=float(np.clip(volume/(a.volume+b.volume-volume),0,1))
    except (ValueError,QhullError,ZeroDivisionError):pass
    return result
