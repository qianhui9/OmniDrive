"""Current native local probe plus read-only metadata-check overhead measurement."""
import ast,json,sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from tools.worldstream_m2 import state_interface_integration as entry
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.camera_executor import numeric_details
root=Path('outputs/m1_m2_window_reuse_20261021_run01')
original_load=entry.load

def historical_method(path,name):
 tree=ast.parse(path.read_text())
 method=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
 scope=dict(torch=torch,numeric_details=numeric_details)
 exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
 return scope[name]

def load():
 model,dataset=original_load()
 rt=IncrementalVisionRuntime(model.img_backbone,dict(mode='exact',cache_capture='always',state_interface='source_linked'))
 assert rt.prefix==[0,1],('Original EVA prefix rejected',rt.planner.rejection_reason)
 old=root/'natural_delivery/source/worldstream/m2'
 old_prefix=historical_method(old/'dependency_planner.py','cacheable_prefix')
 old_signature=historical_method(old/'runtime.py','signature')
 context=dict(execution_signature='metadata_probe')
 result={}
 for name,a,b in [('prefix_validation',lambda:old_prefix(rt.planner),lambda:rt.planner.cacheable_prefix()),
                  ('model_signature',lambda:old_signature(rt,context),lambda:rt.signature(context))]:
  for _ in range(5):a();b()
  values=[[],[]]
  for i in range(100):
   for j in (i%2,1-i%2):
    start=perf_counter();(a,b)[j]();values[j].append((perf_counter()-start)*1000)
  result[name]=dict(archived_p50_ms=float(np.median(values[0])),current_p50_ms=float(np.median(values[1])),
      delta_p50_ms=float(np.median(values[1])-np.median(values[0])),repeats=100)
 result['scope']='Read-only CPU metadata checks on the actual loaded 24-block EVA; outside request timing; does not replace E2E measurements'
 (root/'metadata_guard_overhead.json').write_text(json.dumps(result,indent=2)+'\n')
 return model,dataset

entry.load=load
sys.argv=['state_interface_integration','--window-study','--frames','4','--scenes','1','--repeats','1','--warmup','0',
    '--case','local','--output',str(root/'local_delivery')]
entry.main()
