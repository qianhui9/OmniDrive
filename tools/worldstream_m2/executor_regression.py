"""Isolate compact arithmetic from temporal staleness on real EVA weights/input."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from .common import backbone,write
from .window_validate import difference
from worldstream.m2.dependency_planner import block0_input,DependencyPlanner
from worldstream.m2.packing import partition
from worldstream.m2.window_executor import WindowExecutor,NUMERICAL_POLICY


def run(checkpoint,output):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    cfg.data.test.test_mode=True
    cfg.data.test.pipeline=[n for n in cfg.data.test.pipeline if n['type'] in
        ('LoadMultiViewImageFromFiles','ResizeCropFlipRotImage','ResizeMultiview3D','NormalizeMultiviewImage','PadMultiViewImage')]
    dataset=build_dataset(cfg.data.test);dataset.data_infos.sort(key=lambda i:(i['scene_token'],i['timestamp']))
    image=torch.from_numpy(np.stack(dataset[0]['img']).transpose(0,3,1,2).copy()).cuda()
    model=backbone(checkpoint);executor=WindowExecutor();prefix=DependencyPlanner(model).cacheable_prefix()
    rows=[];inputs=[];outputs=[]
    with torch.no_grad():
        x=block0_input(model,image)
        for i,block in enumerate(model.blocks):
            inputs.append(x);x=block(x);outputs.append(x)
        native=x
        for i in prefix:
            block=model.blocks[i];packed=partition(outputs[i],block.window_size)[0];count=len(packed)
            # Boundary IDs include bottom/right padding in the real 40x40 grid.
            for label,ids in [('none',[]),('one_interior',[0]),('padded_edges',[2,6,8,count-1]),
                              ('half',list(range(0,count,2))),('all',list(range(count)))]:
                actual,_=executor.execute(block,inputs[i],packed,torch.tensor(ids,device=image.device,dtype=torch.long))
                local=difference(actual,outputs[i]);final=actual
                for following in model.blocks[i+1:]:final=following(final)
                rows.append(dict(block=i,case=label,input_shape=list(inputs[i].shape),executed_ids=ids,
                    local=local,final_visual=difference(final,native),finite=bool(torch.isfinite(final).all())))
        if not all(row['finite'] for row in rows):raise ValueError('Nonfinite executor numerical regression')
        if any(not row['local']['exact'] for row in rows if row['case']=='all'):
            raise ValueError('Full-refresh native block arithmetic changed')
    value=dict(frame_id=dataset.data_infos[0]['token'],numerical_policy=NUMERICAL_POLICY,rows=rows,
        executor_sha256=hashlib.sha256(Path('worldstream/m2/window_executor.py').read_bytes()).hexdigest(),
        scope='Fresh exact native cache and unchanged current input: isolates execution rounding, excludes temporal reuse error. No task approval or new tolerance granted.')
    write(out/'results.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();value=run(args.checkpoint,args.output)
    for row in value['rows']:print(row['block'],row['case'],row['local'],row['final_visual'])
