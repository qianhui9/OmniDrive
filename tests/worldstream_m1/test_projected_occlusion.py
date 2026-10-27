from types import SimpleNamespace as NS
import numpy as np
from worldstream.m1.occlusion import projected_occlusion
from worldstream.m1.geometry import intersect_area,region_area


def test_batched_occlusion_matches_original_strict_comparisons(config):
    rng=np.random.RandomState(4)
    boxes=rng.uniform(-50,100,(90,4));boxes[:,2:]=boxes[:,:2]+rng.uniform(0,70,(90,2))
    p={str(i):{'c':NS(region=None if i%9==0 else b.tolist(),depth=float(rng.uniform(0,30)))} for i,b in enumerate(boxes)}
    expected={i:bool(a['c'].region is not None and any(j!=i and b['c'].region is not None and b['c'].depth<a['c'].depth-config.occlusion_depth_margin_m and intersect_area(a['c'].region,b['c'].region)>config.occlusion_overlap_fraction*region_area(a['c'].region) for j,b in p.items())) for i,a in p.items()}
    assert projected_occlusion(p,'c',config)==expected
    assert projected_occlusion({},'c',config)=={}
