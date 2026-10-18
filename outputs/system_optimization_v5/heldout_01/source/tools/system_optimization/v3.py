"""Third-round actual inference, reusing the native runner and bounded worker."""
import copy,gzip,hashlib,json,os,pickle,subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import torch
from . import v2
from .audit import write
from .v2 import Driver as PreviousDriver
from .v2_checks import rows

ROOT=Path('outputs/system_optimization_v3')
MODES=dict(refined_serial='opt_serial',refined_overlap='overlap',certified_serial='opt_serial',
    certified_overlap='overlap',task_full='state_full',task_compact='state_only')


class Driver(PreviousDriver):
    def setup(self,model,mode,*args):
        super().setup(model,MODES.get(mode,mode),*args)
        model.m3_post_verification_ms=0.
        manager=model.m1_manager
        if manager is not None:
            manager.config=replace(manager.config,vectorized_occlusion=mode in ('refined_serial','refined_overlap','certified_serial','certified_overlap','task_full','task_compact'),
                bidirectional_geometry=mode in ('certified_serial','certified_overlap','task_full','task_compact'))
            manager.validator.config=manager.config
            commit=manager.commit
            def capture(frame,outputs,result):
                # Reference only; copies/digests/serialization happen after timer.
                self.current_head_anchors=outputs.anchors
                return commit(frame,outputs,result)
            from functools import wraps
            manager.commit=wraps(commit)(capture)
        if mode in ('task_full','task_compact'):
            model.m3_runtime.config=replace(model.m3_runtime.config,post_perception=True)
        if mode=='unified_auto':
            for attr,name in (('costs','cost_certificate.json'),('quality','quality_certificate.json')):
                setattr(model.stream_runtime,attr,json.loads((ROOT/name).read_text()))

    def before(self,model,mode,rep,index,sink,dest):
        # Frozen source-camera snapshots outside all timing samples.
        self.previous_cameras=copy.deepcopy(model.m1_manager._previous_cameras) if mode=='m1' else None
        self.current_head_anchors=[]
        super().before(model,mode,rep,index,sink,dest)

    def after(self,model,row,sink,dest):
        history=self.old_anchors
        super().after(model,row,sink,dest)
        if row['mode']=='m1' and row['repeat']==self.args.warmup and self.frame is not None:
            p=dest/'oracle_frames'/f'{self.frame.frame_id}.pkl.gz'
            with gzip.open(p,'wb',compresslevel=1) as f:
                pickle.dump(dict(frame=self.frame,history=history,result=model.m1_manager.last_result,
                    masks=model.m1_manager.last_masks,split=row['split'],previous_cameras=self.previous_cameras,
                    temporal_evidence=model.m1_manager.last_evidence,current_head_anchors=self.current_head_anchors),f)
        state=getattr(model,'m3_verified_state',None)
        if row['mode'] in ('task_full','task_compact'):
            row['post_perception']=dict(diagnostics=state.diagnostics if state else None,records=state.records if state else [],
                verification_wall_ms=model.m3_post_verification_ms)


def quality(path):
    data=rows(Path(path)/'frames.jsonl');mapping=dict(opt_serial='m1',refined_serial='m1',overlap='m1',
        refined_overlap='m1',certified_overlap='certified_serial',unified_auto='m1')
    lookup={(r['mode'],r['repeat'],r['index']):r for r in data};pairs=[]
    for r in data:
        mode=r['mode']
        if mode not in mapping or (mapping[mode],r['repeat'],r['index']) not in lookup:continue
        ref=lookup[(mapping[mode],r['repeat'],r['index'])];q=r['frame']['quality']
        p=dict(mode=mode,reference=ref['mode'],index=r['index'],scene=r['frame']['identity']['scene_id'],
            state_equal=q['m1_state_digest']==ref['frame']['quality']['m1_state_digest'],
            token_equal=r['generated_ids']==ref['generated_ids'],
            feature_memory_equal=q['exact_perception_memory_digest'] and all(v['max_abs']==0 and not v['shape_mismatches'] for v in q['numerical'].values()))
        p['passed']=all(p[k] for k in ('state_equal','token_equal','feature_memory_equal'));pairs.append(p)
    value=dict(pairs=pairs,passed=bool(pairs) and all(p['passed'] for p in pairs))
    write(Path(path)/'strict_quality.json',value)
    if pairs and not value['passed']:raise AssertionError('Third-round strict state/output quality failed')
    return value


def arguments(name,kind,start,scenes,frames,modes,warmup=0):
    return v2.arguments(name,kind,start,scenes,frames,modes,warmup=warmup)


def environment():
    return dict(pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),torch=torch.__version__,
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
        threads=len(list(Path('/proc/self/task').iterdir())),
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout,
        gpu_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout)


@torch.no_grad()
def main():
    if set(os.sched_getaffinity(0))&{4,5}:raise RuntimeError('Use taskset -c 0-3,6-31')
    v2.ROOT=ROOT
    import cv2
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False
    write(ROOT/'environment_before.json',environment())
    from tools.worldstream_m2.rearchitecture.integration import load
    model,dataset=load();driver=Driver(model,dataset)
    frozen=json.loads((ROOT/'protocol.json').read_text())
    def run(args):
        driver.run(args);quality(Path(args.output))
    try:
        spec=frozen['runs']['main']
        args=arguments('main','single',spec['scene_start'],spec['scenes'],spec['frames'],
            ['original','m1','refined_serial','overlap','refined_overlap','certified_serial','certified_overlap','m12','task_full','task_compact','m123','native_decode'])
        scenes=sorted({i['scene_token'] for i in dataset.data_infos})[spec['scene_start']:spec['scene_start']+spec['scenes']]
        args.scene_splits={s:'calibration' if i in spec['calibration_scene_ordinals'] else 'heldout' for i,s in enumerate(scenes)}
        run(args)
        from .v3_costs import calibrate
        calibrate(ROOT)
        spec=frozen['runs']['auto']
        run(arguments('auto','single',spec['scene_start'],spec['scenes'],spec['frames'],['m1','unified_auto']))
        spec=frozen['runs']['long']
        run(arguments('long','single',spec['scene_start'],spec['scenes'],spec['frames'],['m1','overlap','refined_overlap','certified_serial','certified_overlap']))
        spec=frozen['runs']['multi']
        run(arguments('multi','multi',spec['scene_start'],spec['scenes'],spec['frames'],['original','prefix','batch','adaptive','task_full','task_compact']))
        from .v3_probes import run as probes
        probes(driver)
        write(ROOT/'environment_after.json',environment())
    finally:driver.close()


if __name__=='__main__':
    if (ROOT/'gpu_status.json').exists():raise RuntimeError('Use a new run root; existing output cannot be overwritten')
    write(ROOT/'gpu_status.json',dict(status='starting'))
    try:main()
    except BaseException as e:
        write(ROOT/'gpu_status.json',dict(status='failed',error=repr(e)));raise
    else:write(ROOT/'gpu_status.json',dict(status='completed'))
