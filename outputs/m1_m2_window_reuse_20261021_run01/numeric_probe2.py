import json,torch
from pathlib import Path
from mmcv import Config
from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT
from worldstream.m2.packing import partition
from worldstream.m2.window_executor import packed_forward
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=dict(Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py').model.img_backbone);cfg.pop('type');cfg.update(depth=2,window_block_indexes=[0,1],with_cp=False)
 m=EVAViT(**cfg).eval();m.load_state_dict(torch.load('outputs/m1_m2_window_reuse_20261021_run01/prefix_weights.pt'),strict=False);m.cuda()
 x=torch.load('outputs/m1_m2_window_reuse_20261021_run01/prefix_input.pt').cuda();block=m.blocks[0];ws=16
 capture={}
 def pre(mod,inp):capture.update(q=inp[0].clone(),kv=inp[1].clone())
 h=block.attn.inner_attn.register_forward_pre_hook(pre)
 ref=block(x);base=dict(capture)
 records=[]
 for n in (1,4,8,16,32,48,53,54):
  windows=partition(x,ws)[0][:n];valid=partition(torch.ones_like(x[...,:1]),ws)[0][:n]
  out=packed_forward(block,windows,valid)
  r=dict(count=n,valid_output=float(((out-partition(ref,ws)[0][:n])*valid).abs().max()))
  for key in ('q','kv'):
   r[key+'_fp32']=float((capture[key]-base[key][:n]).abs().max());r[key+'_fp16_different']=int((capture[key].half()!=base[key][:n].half()).sum())
  records.append(r)
 h.remove();print(json.dumps(records,indent=2),flush=True)
 Path('outputs/m1_m2_window_reuse_20261021_run01/numeric_probe2.json').write_text(json.dumps(records,indent=2))
main()
