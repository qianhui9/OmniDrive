import json
from pathlib import Path
import torch
from tools.worldstream_m2.common import backbone
from worldstream.m2.packing import partition
from worldstream.m2.window_executor import packed_forward
from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos

@torch.no_grad()
def main():
 torch.set_num_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 m=backbone()
 torch.save({k:v.cpu() for k,v in m.state_dict().items() if k.startswith(('patch_embed.','blocks.0.','blocks.1.')) or k=='pos_embed'},'outputs/m1_m2_window_reuse_20261021_run01/prefix_weights.pt')
 from mmcv import Config
 from mmdet3d.datasets import build_dataset
 from mmcv.parallel import collate,scatter
 cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py');cfg.data.test.test_mode=True
 dataset=build_dataset(cfg.data.test)
 batch=scatter(collate([dataset[0]],samples_per_gpu=1),[0])[0]
 images=batch['img'][0].reshape(-1,*batch['img'][0].shape[-3:])
 x=m.patch_embed(images);x=x+get_abs_pos(m.pos_embed,m.pretrain_use_cls_token,x.shape[1:3])
 torch.save(x.cpu(),'outputs/m1_m2_window_reuse_20261021_run01/prefix_input.pt')
 block=m.blocks[0];ws=block.window_size
 records=[]
 def capture(store):
  return [getattr(block,name).register_forward_hook(lambda mod,inp,out,name=name:store.update({name:out.detach().clone()})) for name in ('norm1','attn','norm2','mlp')]
 base={};handles=capture(base);ref=block(x)
 for h in handles:h.remove()
 for count in (1,2,4,8,16,32,54):
  windows=partition(x,ws)[0][:count];valid=partition(torch.ones_like(x[...,:1]),ws)[0][:count]
  actual={};handles=capture(actual);out=packed_forward(block,windows,valid)
  for h in handles:h.remove()
  record=dict(count=count,output=float((out-partition(ref,ws)[0][:count]).abs().max()))
  for name in actual:
   expected=partition(base[name],ws)[0][:count] if name!='attn' else base[name][:count]
   record[name]=float((actual[name]-expected).abs().max())
  records.append(record)
 print(json.dumps(records,indent=2),flush=True)
 Path('outputs/m1_m2_window_reuse_20261021_run01/numeric_probe.json').write_text(json.dumps(records,indent=2))
main()
