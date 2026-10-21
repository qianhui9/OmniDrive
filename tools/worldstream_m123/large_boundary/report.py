"""Measured coarse scope, dependency rejections and deliberately generous bounds."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

from tools.worldstream_m123.stateful_opportunity.common import manifest,protected_files,sha256,stats,write
from tools.worldstream_m123.stateful_opportunity.report import opportunity_gate


def read(root,name):return json.loads((root/name).read_text())


def removal_bound(total,removable,mandatory):
    """Zero-cost scope oracle; preserve the actual input pipeline lower bound.

    Removable is an intentionally loose CPU+GPU envelope, not a measured
    speedup. Mandatory input preprocessing/scatter cannot disappear in this task.
    """
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (total,removable,mandatory)):
        raise ValueError('Unmeasured bounds cannot be replaced by zero')
    if not 0<mandatory<=total or removable<0:raise ValueError('Invalid service envelope')
    saving=min(removable,total-mandatory)
    return dict(removable_upper_ms=saving,speedup_upper=total/(total-saving))


def scope_sum(frame,names,field):
    scopes=frame['scopes']['inclusive_module_intervals']
    return sum(scopes[n][field] for n in names)
