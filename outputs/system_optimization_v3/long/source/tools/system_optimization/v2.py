"""Second-round experiments reuse the original native runner and contracts."""
import argparse,ast,copy,gzip,hashlib,importlib,json,pickle,subprocess,types
from dataclasses import fields,is_dataclass,replace
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
from .audit import write
from . import experiment as experiment

ROOT=Path('outputs/system_optimization_v2')
ALIASES=dict(opt_serial='m1',overlap='m1',overlap_decode='m1',native_decode='original',
    opt_m12='m12',original_m3='m3',unified_auto='m1')


def digest(value):
    h=hashlib.sha256()
    def add(x):
        if is_dataclass(x):return add({f.name:getattr(x,f.name) for f in fields(x)})
        if isinstance(x,torch.Tensor):return add(x.detach().cpu().numpy())
        if isinstance(x,np.ndarray):h.update(repr((x.shape,str(x.dtype))).encode());h.update(x.tobytes());return
        if isinstance(x,dict):
            for k,v in sorted(x.items()):
                if k=='timings_ms' or k.endswith('_ms'):continue
                h.update(str(k).encode());add(v)
        elif isinstance(x,(tuple,list)):
            for v in x:add(v)
        else:h.update(repr(x).encode())
    add(value);return h.hexdigest()


def state_digest(manager):
    if manager is None:return None
    return digest(dict(anchors=manager.store.anchors,features=manager.store.features,
        provenance=manager.store.provenance,precheck=manager.last_result,masks=manager.last_masks,
        evidence=manager.last_evidence,camera_history=manager._previous_cameras,
        epoch=manager.store.generation,commits=manager.store.commit_count))


def original_entry(model):
    path=ROOT/'baseline_source/petr3d.py'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(subprocess.check_output(['git','show','b4782abfa9d8f1e5ff5980a1ce484777ea55ce3f:projects/mmdet3d_plugin/models/detectors/petr3d.py']))
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Petr3D')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='simple_test')
    module=importlib.import_module(type(model).__module__);ns=dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),ns)
    return ns['simple_test']


def arguments(output,workload,scene_start,scenes,frames,modes,warmup=1,repeats=1,stride=1,refresh=10):
    return argparse.Namespace(output=str(ROOT/output),workload=workload,scene_start=scene_start,scenes=scenes,
        frames=frames,modes=modes,warmup=warmup,repeats=repeats,frame_stride=stride,cache_budget_mb=512,
        refresh_interval=refresh,profile_stages=True,trace=False,decode_probe=False,lifecycle_probe=False)


class Driver:
    def __init__(self,model,dataset):
        from worldstream.m1 import omnidrive_adapter as adapter
        self.model,self.dataset=model,dataset
        self.native_generate=model.lm_head.generate
        self.native_m3=experiment.RecordedM3.generate
        self.configure=experiment.configure
        self.new_simple=type(model).simple_test;self.old_simple=original_entry(model)
        self.context=None;self.frame=None;self.old_anchors=None;self.args=None
        self.weight_hash=json.loads((ROOT/'model_manifest.json').read_text())['sha256']
        self.adapter=adapter;self.original_adapter=adapter.frame_from_model
        def capture(*a,**kw):
            frame=self.original_adapter(*a,**kw);self.frame=frame;return frame
        adapter.frame_from_model=capture

    def setup(self,model,mode,sink,dest,budget,refresh,profile):
        if self.context is not None:self.context.__exit__(None,None,None);self.context=None
        self.configure(model,ALIASES.get(mode,mode),sink,dest,budget,refresh,profile)
        old=mode in ('m1','m12','m123','original_m3')
        model.simple_test=types.MethodType(self.old_simple if old else self.new_simple,model)
        config=model.stream_runtime.config
        model.stream_runtime.config=replace(config,reuse_m1_signature=mode in ('opt_serial','overlap','overlap_decode','opt_m12','unified_auto'),
            m1_overlap='experimental' if mode in ('overlap','overlap_decode') else 'auto' if mode=='unified_auto' else 'serial',
            cost_aware=mode=='unified_auto',capture_cost_key=mode in ('overlap','overlap_decode','unified_auto'),
            generation_key='native_sampling_temp0.1_topp0.75_beams1_max320_cache' if self.args.workload=='single' else 'native_greedy_beams1_max320_cache',
            model_version='iter_10548.pth@sha256:'+self.weight_hash,lora_version='embedded-original@sha256:'+self.weight_hash)
        if mode=='unified_auto':
            for attr,file in (('costs','cost_certificate.json'),('quality','quality_certificate.json')):
                path=ROOT/file
                if path.exists():setattr(model.stream_runtime,attr,json.loads(path.read_text()))
        if mode in ('native_decode','overlap_decode'):
            from worldstream.m3.native_decode import zero_mask_decode
            self.context=zero_mask_decode(model.lm_head);self.decode_stats=self.context.__enter__()
        self.frame=None

    def before(self,model,mode,rep,index,sink,dest):
        self.frame=None
        self.old_anchors=copy.deepcopy(model.m1_manager.store.anchors) if mode=='m1' and rep==self.args.warmup else None

    def after(self,model,row,sink,dest):
        row['frame']['quality']['m1_state_digest']=state_digest(model.m1_manager)
        if self.context is not None:row['frame']['m3']['native_decode_stats']=dict(self.decode_stats)
        row['dataset_identity']=dict(sample_token=self.dataset.data_infos[row['index']]['token'],
            scene_token=self.dataset.data_infos[row['index']]['scene_token'],timestamp=self.dataset.data_infos[row['index']]['timestamp']/1e6)
        info=row['dataset_identity'];actual=row['frame']['identity']
        if info['sample_token']!=actual['frame_id'] or info['scene_token']!=actual['scene_id'] or abs(info['timestamp']-actual['timestamp'])>1e-5:
            raise ValueError('Actual model execution differs from stable Dataset identity')
        if row['mode']=='m1' and row['repeat']==self.args.warmup and self.frame is not None:
            path=dest/'oracle_frames'/f'{self.frame.frame_id}.pkl.gz';path.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(path,'wb',compresslevel=1) as handle:
                pickle.dump(dict(frame=self.frame,history=self.old_anchors,result=model.m1_manager.last_result,
                    masks=model.m1_manager.last_masks,split=row['split']),handle)
        self.old_anchors=None

    def run(self,args):
        from tools.worldstream_m2.rearchitecture import integration
        self.args=args;self.model.lm_head.generate=self.native_generate
        experiment.RecordedM3.generate=self.native_m3
        experiment.configure=self.setup
        experiment.RUN_HOOKS.update(before=self.before,after=self.after)
        load=integration.load;integration.load=lambda:(self.model,self.dataset)
        try:
            experiment.run(args)
        except Exception as error:
            path=Path(args.output)
            if path.exists():write(path/'status.json',dict(status='failed',error=repr(error)))
            raise
        finally:
            integration.load=load
            if self.context is not None:self.context.__exit__(None,None,None);self.context=None
            self.model.stream_runtime.close()

    def close(self):
        self.adapter.frame_from_model=self.original_adapter
        experiment.configure=self.configure;experiment.RUN_HOOKS.clear()
        experiment.RecordedM3.generate=self.native_m3
        self.model.lm_head.generate=self.native_generate


@torch.no_grad()
def main(stage):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    from tools.worldstream_m2.rearchitecture.integration import load
    model,dataset=load();driver=Driver(model,dataset)
    try:
        if stage=='smoke':
            runs=[arguments('smoke','single',100,2,2,['original','m1','opt_serial','overlap','native_decode','overlap_decode'],warmup=0)]
        else:
            runs=[arguments('single','single',102,4,4,['original','m1','opt_serial','overlap','native_decode','overlap_decode','m12','opt_m12','original_m3','m123','unified_auto']),
                arguments('multi','multi',102,4,2,['original','prefix','batch','adaptive','state_only','state_full','original_m3','m123','unified_auto']),
                arguments('long_stream','single',106,4,32,['original','m1','overlap'],warmup=0),
                arguments('stride_refresh','single',110,2,4,['original','m1','overlap'],stride=2,refresh=2)]
        from .v2_checks import check
        for args in runs:
            driver.run(args)
            if 'overlap' in args.modes:check(args.output)
        if stage=='formal':
            from .v2_costs import calibrate
            calibrate(ROOT)
            # Fresh independent scenes verify runtime decisions after freezing
            # certificates. Failed or missing keys must retain full serial M1.
            driver.run(arguments('auto_validation','single',112,2,4,['m1','unified_auto'],warmup=0))
            check(ROOT/'auto_validation')
        from .v2_probes import run as probes
        driver.args=runs[0]
        probes(driver,ROOT/('smoke' if stage=='smoke' else 'single'),ROOT/(stage+'_probes'))
        if stage=='formal':
            from .v2_batch_probe import run as batch_probe
            batch_probe(driver,ROOT/'batch_probes')
    finally:driver.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','formal'],default='smoke')
    stage=p.parse_args().stage;status=ROOT/(stage+'_status.json')
    write(status,dict(status='starting'))
    try:main(stage)
    except Exception as error:
        write(status,dict(status='failed',error=repr(error)));raise
    else:write(status,dict(status='completed',all_stage_runs_and_probes_completed=True))
