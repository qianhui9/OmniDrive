"""Actual preprocessed frame and executed window footprints; no GT identity claim."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin

ROOT=Path(__file__).resolve().parent
TOKEN='a7795dcadad44c1c85fa27eef18366f6'
def frame(path):
    with path.open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['frame_id']==TOKEN:return row['m2']
    raise ValueError('Missing actual anomaly frame')
old=frame(ROOT/'references/wsgrrr_20261022_followup/D3_full/per_frame_metrics.jsonl')
new=frame(ROOT/'main_full/per_frame_metrics.jsonl')
cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
cfg.data.test.test_mode=True
cfg.data.test.pipeline=[node for node in cfg.data.test.pipeline if node['type'] in
    ('LoadMultiViewImageFromFiles','ResizeCropFlipRotImage','ResizeMultiview3D','NormalizeMultiviewImage','PadMultiViewImage')]
dataset=build_dataset(cfg.data.test)
index=next(i for i,info in enumerate(dataset.data_infos) if info['token']==TOKEN)
sample=dataset[index];info=dataset.data_infos[index]
camera=list(info['cams']).index('CAM_FRONT')
norm=sample['img_norm_cfg'];rgb=sample['img'][camera]*np.asarray(norm['std'])+np.asarray(norm['mean'])
if not norm.get('to_rgb'):rgb=rgb[...,::-1]
rgb=np.clip(rgb,0,255).astype(np.uint8)
box=next(r['bbox_xyxy'] for r in new['region_refresh']['uncertainty_regions']
    if r['camera_id']=='CAM_FRONT' and r['source_track_id'].endswith(':a4') and 4 in r['windows'])
fig,axes=plt.subplots(1,2,figsize=(12,6))
colors=dict(reuse='#607d8b',ordinary='#ffb300',mandatory='#e53935')
for ax,diag,title in zip(axes,[old,new],['Before: window 4 reused','Main fix: window 4 mandatory']):
    ax.imshow(rgb);ax.set_title(title)
    selected=set(diag['stages'][0]['executed_window_ids'])
    required=set(diag['region_refresh']['mandatory_window_ids'])
    footprints=diag['prefix_dependency_footprints']['0']
    offset=camera*len(footprints)
    for j,item in enumerate(footprints):
        x0,y0,x1,y1=item['input_region'];ident=offset+j
        kind='mandatory' if ident in required else 'ordinary' if ident in selected else 'reuse'
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=colors[kind],linewidth=2.5))
        ax.text(x0+6,y0+19,str(ident),color='white',fontsize=11,
            bbox=dict(facecolor=colors[kind],edgecolor='none',alpha=.9))
    x0,y0,x1,y1=box
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#00e5ff',linewidth=2))
    ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0);ax.set_axis_off()
fig.legend(handles=[Patch(facecolor=v,label=k) for k,v in colors.items()]+[
    Patch(facecolor='#00e5ff',label='M1 image risk box (not certified world identity)')],
    loc='lower center',ncol=4,fontsize=9)
fig.suptitle('CAM_FRONT | '+TOKEN+' | actual Conv2d window footprints',fontsize=12)
fig.tight_layout(rect=(0,.06,1,.95))
fig.savefig(ROOT/'anomaly_front_window4.png',dpi=160)
plt.close(fig)
