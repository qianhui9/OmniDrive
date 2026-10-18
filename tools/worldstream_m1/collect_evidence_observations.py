"""Original checkpoint detections on a previously frozen scene list. No GT read."""
import hashlib
import json
from pathlib import Path
import pickle
import random
from time import perf_counter
import numpy as np
from .io_utils import write_json


def main(root=None):
    import torch
    from mmcv import Config
    from mmcv.parallel import collate,scatter
    from mmcv.runner import load_checkpoint,wrap_fp16_model
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    from worldstream.m1 import M1Config
    from worldstream.m1.omnidrive_adapter import outputs_from_model
    from .dataset import frame_from_info
    root=Path(root or 'outputs/module1_evidence_optimization');target=root/'model_observations.pkl'
    if target.exists():raise RuntimeError('Preserve existing model observations; refusing overwrite')
    selection=json.loads((root/'data_selection.json').read_text())
    if not torch.cuda.is_available():raise RuntimeError('Host GPU access required')
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    cfg.model.update(m1_enable=False,m2_enable=False,m3_enable=False,lm_head=None,train_cfg=None,pretrained=None)
    cfg.data.test.ann_file=selection['info_file'];cfg.data.test.test_mode=True
    dataset=build_dataset(cfg.data.test);by_token={i['token']:j for j,i in enumerate(dataset.data_infos)}
    # Dataset test_mode excludes label-loading transforms. Input preprocessing,
    # checkpoint, score/NMS decoding and all model inference settings are intact.
    model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):wrap_fp16_model(model)
    load_checkpoint(model,'ckpts/iter_10548.pth',map_location='cpu');model.cuda().eval()
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    settings=M1Config();outputs={};times=[];frame_contracts=[]
    with torch.no_grad():
        for split,specs in selection['splits'].items():
            for spec in specs:
                model.test_flag=False
                for token in spec['samples']:
                    i=by_token[token];info=dataset.data_infos[i]
                    batch=scatter(collate([dataset[i]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                    torch.cuda.synchronize();tick=perf_counter()
                    result=model(return_loss=False,rescale=True,**batch)
                    torch.cuda.synchronize();times.append((perf_counter()-tick)*1000)
                    # Only known input pose and actual prediction tensors are
                    # passed to the existing read-only output adapter.
                    frame=frame_from_info(info,'data/nuscenes')
                    out=outputs_from_model(frame,[result[0]['pts_bbox']],None,{},settings)
                    out.source='historical_full_model_detection_replay'
                    outputs[token]=out
                    frame_contracts.append(dict(scene=spec['scene_id'],sample=token,timestamp=frame.timestamp,
                                                camera_order=list(frame.cameras),split=split))
                print(split,spec['scene_id'],'actual perception frames',len(spec['samples']),flush=True)
    with target.open('wb') as handle:pickle.dump(outputs,handle,protocol=4)
    write_json(root/'model_observation_provenance.json',dict(source='New actual original checkpoint perception outputs, replayed later on CPU',
        selection_sha256=hashlib.sha256((root/'data_selection.json').read_bytes()).hexdigest(),
        prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),frames=frame_contracts,
        perception_wall_ms=times,torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),model_parameters_changed=False,model_train_scenes=True,
        label_inputs_used=False,language_generation=False,M2_M3_enabled=False))


if __name__=='__main__':main()
