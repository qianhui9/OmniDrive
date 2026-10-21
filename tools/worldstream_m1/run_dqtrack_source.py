"""Standalone, small DETR3D DQTrack check/export using an external source tree.

Run this script in the DQTrack interpreter, never import it into the OmniDrive
model process. No dependency installation or upstream source modification.
Selective registry imports avoid the unrelated stereo CUDA extension.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import pickle
import pickletools
import zipfile
from pathlib import Path
import runpy
import sys
import os
import subprocess
import time
import types
import uuid


def load_detr3d_plugin(root):
    if 'projects' in sys.modules:
        raise RuntimeError('Use a fresh process: projects namespace already loaded')
    sys.dont_write_bytecode=True
    root = Path(root).resolve()
    prefix = 'projects.mmdet3d_plugin'
    packages = ['projects', prefix] + [prefix+'.'+name for name in (
        'core','core.bbox','core.bbox.assigners','core.bbox.coders','core.bbox.match_costs',
        'core.track','models','models.utils','models.trackers','models.track_head',
        'models.losses','datasets','datasets.pipelines')]
    for name in packages:
        path = root.joinpath(*name.split('.'))
        if not path.is_dir():
            raise FileNotFoundError(path)
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    modules = (
        'core.bbox.assigners.hungarian_assigner_3d', 'core.bbox.coders.nms_free_coder',
        'core.bbox.coders.detr_track_coder', 'core.bbox.match_costs.match_cost',
        'models.utils.detr3d_transformer','models.losses.categorical_cross_entroy',
        'models.losses.dual_query_matcher','models.track_head.detr3d_head',
        'models.trackers.detr3d_tracker_dq','datasets.nuscenes_dataset_track',
        'datasets.pipelines.loading_3d','datasets.pipelines.formatting',
        'datasets.pipelines.transform_3d')
    for name in modules:
        importlib.import_module(prefix+'.'+name)
    return modules


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()



class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','OrderedDict'),('torch._utils','_rebuild_tensor_v2'),
                 ('torch','FloatStorage'),('torch','LongStorage')}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError('Rejected checkpoint global: '+module+'.'+name)
        return super().find_class(module,name)


class _TensorPickle:
    __name__='dqtrack_restricted_pickle'
    Unpickler=_CheckpointUnpickler


def load_tensor_checkpoint(path):
    """Restricted tensor+primitive loader for the audited legacy checkpoint.

    Torch 1.13 weights_only rejects BINFLOAT in this checkpoint's metadata.
    This loader permits only its four inspected tensor globals, never an
    unrestricted pickle fallback. TorchScript and pickle extension codes reject.
    """
    import torch
    with zipfile.ZipFile(path) as archive:
        if any(n.endswith('/constants.pkl') for n in archive.namelist()):
            raise ValueError('TorchScript archive is not a tensor checkpoint')
        names=[n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(names)!=1:raise ValueError('Expected one tensor pickle')
        for op,arg,_ in pickletools.genops(archive.read(names[0])):
            if op.name in ('EXT1','EXT2','EXT4','STACK_GLOBAL'):
                raise ValueError('Dynamic pickle globals are not allowed')
            if op.name=='GLOBAL' and arg not in (
                    'collections OrderedDict','torch._utils _rebuild_tensor_v2',
                    'torch FloatStorage','torch LongStorage'):
                raise pickle.UnpicklingError('Rejected checkpoint global: '+arg)
    # Torch 1.13 torch.load overwrites an explicit pickle_module when weights_only
    # is false. Call its tensor archive reader with the restricted module directly.
    with torch.serialization._open_zipfile_reader(str(path)) as archive:
        return torch.serialization._load(archive,map_location='cpu',pickle_module=_TensorPickle)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True,help='New output directory; never overwritten')
    p.add_argument('--config',default='projects/configs/dqtrack/detr3d_r101_size_900_1600_dq.py')
    p.add_argument('--run',action='store_true',help='Otherwise CPU strict-load check only')
    p.add_argument('--check-data',action='store_true',help='Also validate selected data through CPU pipeline without forward')
    p.add_argument('--gpu-wait-seconds',type=float,default=60)
    p.add_argument('--infos')
    p.add_argument('--data-root')
    p.add_argument('--scene')
    p.add_argument('--scene-list',help='Frozen protocol JSON: development entries with scene_id and samples')
    p.add_argument('--frames',type=int,default=2)
    a=p.parse_args()
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status='started',actual_inference=False,inference_attempted=False,run_id=str(uuid.uuid4()),source_root=str(Path(a.source_root).resolve()),
                arguments=vars(a))
    try:
        import torch
        from mmcv import Config
        from mmdet3d.models import build_model
        cfg_path=Path(a.source_root)/a.config
        # Supplied project configs are executable trusted local Python, like MMCV configs.
        values=runpy.run_path(str(cfg_path))
        cfg=Config({k:v for k,v in values.items() if not k.startswith('_')})
        if cfg.model.type!='DETR3DTrackerDQ' or cfg.model.tracker_cfg.eval_det_only:
            raise ValueError('This audited entry supports DETR3D tracking only')
        report.update(config_sha256=file_sha(cfg_path),checkpoint_sha256=file_sha(a.checkpoint),
                      resolved_model=cfg.model.to_dict(),torch_version=torch.__version__)
        report['source_sha256']={str(p.relative_to(a.source_root)):file_sha(p)
            for p in sorted(Path(a.source_root).rglob('*.py'))}
        report['selected_imports']=load_detr3d_plugin(a.source_root)
        cfg.model.pretrained=None
        if 'init_cfg' in cfg.model.img_backbone:cfg.model.img_backbone.init_cfg=None
        # Upstream allocates a parameter-free lookup on CUDA in its constructor.
        # Construct it on CPU; only move it to CUDA when an actual run is allowed.
        from projects.mmdet3d_plugin.models.trackers.detr3d_tracker_dq import DETR3DTrackerDQ
        original=DETR3DTrackerDQ.trans_track_cate
        def cpu_categories(self):
            classes=self.tracker_cfg.class_dict
            return torch.tensor([classes.all_classes.index(name) for name in classes.track_classes],dtype=torch.long)
        DETR3DTrackerDQ.trans_track_cate=cpu_categories
        report['cpu_check_override']='category lookup allocation device only; no parameter changes'
        try:model=build_model(cfg.model,train_cfg=None,test_cfg=cfg.get('test_cfg'))
        finally:DETR3DTrackerDQ.trans_track_cate=original
        checkpoint=load_tensor_checkpoint(a.checkpoint)
        report['checkpoint_loader']='restricted tensor/primitive pickle; four allowed globals'
        weights=checkpoint.get('state_dict',checkpoint)
        # No aliases, prefix stripping, omitted tensors or non-strict fallback.
        model.load_state_dict(weights,strict=True)
        report.update(status='strict_load_passed',parameter_tensors=len(weights))
        model.eval()
        if a.run or a.check_data:
            if not all((a.infos,a.data_root,a.scene or a.scene_list)) or not 1<=a.frames<=40:
                raise ValueError('Run needs infos/data-root/scene and 1..40 frames')
            if a.scene_list:
                protocol=json.loads(Path(a.scene_list).read_text())
                specs=protocol['development']
                if not 1<=len(specs)<=8 or sum(len(s['samples']) for s in specs)>320:
                    raise ValueError('Explicit development budget: <=8 scenes and <=320 frames')
                if len({s['scene_id'] for s in specs})!=len(specs):raise ValueError('Duplicate scenes')
                scene_reports=[]
                for spec in specs:
                    if not 1<=len(spec['samples'])<=80:raise ValueError('Bounded continuous scenes only')
                    import copy
                    sub=copy.copy(a);sub.scene=spec['scene_id'];sub.frames=len(spec['samples'])
                    sub.expected_tokens=spec['samples']
                    folder=output/sub.scene;folder.mkdir()
                    item=dict(actual_inference=False,run_id=report['run_id'],checkpoint_sha256=report['checkpoint_sha256'])
                    run_export(model,cfg,sub,folder,item)
                    (folder/'run_report.json').write_text(json.dumps(item,indent=2))
                    scene_reports.append(dict(scene_id=sub.scene,**item))
                    print('exported',sub.scene,len(item.get('frames',[])),flush=True)
                report.update(status='multiscene_exported',actual_inference=a.run,scenes=scene_reports)
            else:run_export(model,cfg,a,output,report)
    except Exception as exc:
        import traceback
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        (output/'run_report.json').write_text(json.dumps(report,indent=2,default=str))


def wait_for_gpu(timeout):
    if os.environ.get('CUDA_VISIBLE_DEVICES','0')!='0':
        raise ValueError('This small runner supports physical GPU 0 only')
    if not 0<=timeout<=3600:raise ValueError('GPU wait must be 0..3600 seconds')
    deadline=time.monotonic()+timeout
    while True:
        processes=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'],text=True).strip()
        if not processes or all(row.strip()==str(os.getpid()) for row in processes.splitlines()):return
        print('GPU 0 occupied; waiting without stopping other processes',flush=True)
        if time.monotonic()>=deadline:raise TimeoutError('GPU occupied; no inference started')
        time.sleep(min(10,max(0,deadline-time.monotonic())))


def run_export(model,cfg,args,output,report):
    # Imports stay inside this separate process. Dataset uses trusted local pickle.
    import time
    import torch
    import numpy as np
    import mmcv
    from mmcv.parallel import MMDataParallel
    from mmdet3d.datasets import build_dataset,build_dataloader
    from nuscenes.eval.detection.config import config_factory
    from mmcv.runner import wrap_fp16_model
    if float(importlib.import_module('mmdet3d').__version__[:3])<1.0:
        raise ValueError('Export requires verified >=1.0 MMDetection3D box convention')
    report['infos_sha256']=file_sha(args.infos)
    data_cfg=cfg.data.test.copy()
    data_cfg.update(data_root=str(Path(args.data_root).resolve()),ann_file=str(Path(args.infos).resolve()),test_mode=True)
    if not hasattr(model,'_dqtrack_replay_dataset'):
        model._dqtrack_replay_dataset=build_dataset(data_cfg)
        model._dqtrack_all_infos=list(model._dqtrack_replay_dataset.data_infos)
    dataset=model._dqtrack_replay_dataset
    dataset.data_infos=list(model._dqtrack_all_infos)
    sequence=sorted((i for i in dataset.data_infos if i['scene_token']==args.scene),key=lambda i:i['timestamp'])
    if not sequence:raise ValueError('Scene absent from official infos')
    dataset.data_infos=sequence[:args.frames]
    if hasattr(args,'expected_tokens') and [i['token'] for i in dataset.data_infos]!=args.expected_tokens:
        raise ValueError('Frozen sample sequence disagrees with official infos')
    samples=json.loads((Path(args.data_root)/'v1.0-trainval/sample.json').read_text())
    samples={row['token']:row for row in samples}
    for index,info in enumerate(dataset.data_infos):
        sample=samples[info['token']]
        if sample['scene_token']!=info['scene_token'] or sample['timestamp']!=info['timestamp']:
            raise ValueError('Official infos disagree with sample.json frame metadata')
        expected_prev=dataset.data_infos[index-1]['token'] if index else ''
        if sample['prev']!=expected_prev:
            raise ValueError('Export must use a continuous prefix from the scene start')
    dataset.flag=np.zeros(len(dataset),dtype=np.uint8)
    # Resolve original converter's data/nuscenes paths without editing original infos.
    root=Path(args.data_root).resolve()
    def resolve(path):
        parts=Path(path).parts
        for marker in ('samples','sweeps'):
            if marker in parts:return str(root.joinpath(*parts[parts.index(marker):]))
        raise ValueError('Unrecognized nuScenes path: '+str(path))
    for info in dataset.data_infos:
        for cam in info['cams'].values():cam['data_path']=resolve(cam['data_path'])
        for sweeps in info.get('cam_sweeps_info',{}).values():
            for sweep in sweeps:sweep['data_path']=resolve(sweep['data_path'])
    loader=build_dataloader(dataset,samples_per_gpu=1,workers_per_gpu=0,dist=False,shuffle=False)
    if not args.run:
        shapes=[]
        for data in loader:
            container=data['img']
            shapes.append(str(getattr(container,'data',container)[0].shape))
        report.update(status='cpu_data_checked',data_frames=len(dataset),input_shapes=shapes,
                      sample_tokens=[i['token'] for i in dataset.data_infos])
        return
    wait_for_gpu(args.gpu_wait_seconds)
    if cfg.get('fp16'):wrap_fp16_model(model)
    # Explicit reset for every scene, including repeated runs of the same scene.
    model.test_detector=None;model.test_tracker=None;model.scene_token=None
    model.max_track_id=0
    model.det2track=model.det2track.cuda()
    torch.cuda.reset_peak_memory_stats()
    model=MMDataParallel(model.cuda(),device_ids=[0])
    # Load the dependency-light exporter without importing OmniDrive's projects namespace.
    path=Path(__file__).resolve().parents[2]/'worldstream/m1/dqtrack_export.py'
    spec=importlib.util.spec_from_file_location('m1_dqtrack_export',path)
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    results={};frames={};counts=[]
    for index,data in enumerate(loader):
        info=dataset.data_infos[index];torch.cuda.synchronize();start=time.perf_counter()
        report['inference_attempted']=True
        with torch.no_grad():prediction=model(return_loss=False,rescale=True,**data)
        report['actual_inference']=True
        torch.cuda.synchronize()
        forward_ms=(time.perf_counter()-start)*1000
        if len(prediction)!=1:raise ValueError('One chronological sample per call required')
        rows,diagnostics=exporter.export_frame(prediction[0],info,dataset.track_classes,
                                               config_factory('detection_cvpr_2019').class_range)
        token=info['token'];results[token]=rows
        frames[token]=dict(sample_token=token,scene_id=info['scene_token'],timestamp=info['timestamp']/1e6)
        counts.append(dict(sample_token=token,forward_ms=forward_ms,elapsed_ms=(time.perf_counter()-start)*1000,**diagnostics))
    payload=dict(results=results,m1_manifest=dict(run_id=report['run_id'],coordinate_system='world',
        size_order='wlh',timestamp_unit='seconds',model_config=str(Path(args.source_root)/args.config),
        checkpoint_sha256=report['checkpoint_sha256'],frames=frames))
    (output/'normalized_tracking.json').write_text(json.dumps(payload,allow_nan=False))
    report.update(status='exported',actual_inference=True,frames=counts,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  data_split='exposed_development_smoke',independent_validation=False)


if __name__=='__main__':main()
