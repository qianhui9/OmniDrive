import sys,json
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager,FullOutputs,PerceptionAnchor,Frame,CameraFrame
from worldstream.m1.geometry import box_geometry
m=WorldStateManager(M1Config())
for t in (100.,100.5):
 image=np.random.RandomState(9).randint(0,256,(100,120,3),dtype=np.uint8)
 k=np.array([[60.,0,60],[0,60,50],[0,0,1]])
 mask=np.ones((100,120),bool)
 cam=CameraFrame('front',image,k,np.eye(4),t,visibility_mask=mask,geometry_support_mask=mask)
 f=Frame('scene',str(t),t,np.eye(4),{'front':cam})
 r=m.precheck(f);m.commit(f,FullOutputs([PerceptionAnchor('object','car',box_geometry([0,0,10],[2,2,2]))]),r)
a=m.export_state_audit()
# Compare every state/provenance field except wall-clock measurements.
def strip(x):
 if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if not k.endswith('_ms') and k!='timings_ms'}
 if isinstance(x,list):return [strip(v) for v in x]
 return x
with open(sys.argv[1],'x') as h:json.dump(strip(a),h,sort_keys=True)
