"""Audit R0 retirement and compare real pre/post cleanup A/C receipts."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from .benchmark.common import sha256, write
from .source_provenance import historical_source

ROOT=Path('outputs/r0_route_cleanup_validation_run01')


def read(path):
    return json.loads(Path(path).read_text())


def subset(after,before):
    """Copy the exact predeclared frame/question selection, never outcomes."""
    protocol=read(after/'experiment_protocol.json')
    for name,key in [('scene_selection.json','selected_scenes_sha256'),
                     ('workload_manifest.json','workloads_sha256'),
                     ('data_role_manifest.json','roles_sha256'),
                     ('scene_condition_manifest.json','conditions_sha256')]:
        (after/name).write_bytes((before/name).read_bytes())
        protocol[key]=sha256(after/name)
    selected=read(after/'scene_selection.json')['selected']
    count=sum(len(s['tokens']) for s in selected)
    protocol.update(frames_per_arm=count*5,planned_main_model_calls=count*10,
        validation_scope='PREDECLARED_BEFORE_AFTER_CLEANUP_EQUIVALENCE_TWO_COMPLETE_LOGS')
    write(after/'experiment_protocol.json',protocol)


def prepare(root):
    from .frozen_main_benchmark import prepare as freeze
    after=root/'after'
    freeze(after)
    subset(after,root/'before')
    print('AFTER_CLEANUP_PROTOCOL_FROZEN',after)


def audit(root):
    from worldstream.r0 import model_settings, m3_settings
    from worldstream.m3 import M3Config
    from dataclasses import asdict
    before=read(root/'pre_cleanup_sources.json')
    reference=read(root/'frozen_semantic_parameters.json')
    assert model_settings()==reference
    assert asdict(M3Config())==reference['m3_config']
    removed=[p for p in before if not Path(p).exists()]
    changed=[p for p,h in before.items() if Path(p).is_file() and sha256(p)!=h]
    # Imports are resolved against previous tracked module names, including
    # package __init__.py. The inference package must not depend on retired code.
    modules={p[:-3].replace('/','.').removesuffix('.__init__'):p for p in before if p.endswith('.py')}
    broken=[];compiled=0
    candidates={Path(p) for p in before if p.endswith('.py') and Path(p).is_file()}
    for base in ['worldstream','tools','tests','projects']:
        candidates.update(Path(base).rglob('*.py'))
    for p in sorted(candidates):
        t=ast.parse(p.read_text());compile(t,str(p),'exec');compiled+=1
        mod=str(p.with_suffix('')).replace('/','.').removesuffix('.__init__')
        for n in ast.walk(t):
            names=[]
            if isinstance(n,ast.Import):names=[a.name for a in n.names]
            if isinstance(n,ast.ImportFrom):
                pkg=mod.split('.') if p.name=='__init__.py' else mod.split('.')[:-1]
                pre='.'.join(pkg[:len(pkg)-n.level+1]) if n.level else ''
                name='.'.join(x for x in [pre,n.module or ''] if x)
                names=[name]+[name+'.'+a.name for a in n.names]
            for name in names:
                if name in modules and not Path(modules[name]).is_file():broken.append([str(p),name])
    assert not broken,broken
    # The learned task heads, vision model and native LLM generation expression
    # remain unchanged. Kept numerical M2 methods are compared as ASTs.
    baseline=read(root/'baseline_metadata.json')['commit']
    def old(path):return ast.parse(historical_source(baseline,path).decode())
    def node(t,name):return next(n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    proofs={}
    for path,names in [('worldstream/m3/prefill_executor.py',['GenerationFacade','deterministic_draft_sample']),
                       ('worldstream/m3/runtime.py',['_decode_binding','validate_contract']),
                       ('worldstream/m2/runtime.py',['_forward','_forward_frame','_plan_refresh','forward_from_m1','signature'])]:
        a=old(path);b=ast.parse(Path(path).read_text())
        for name in names:proofs[path+':'+name]=ast.dump(node(a,name))==ast.dump(node(b,name))
    p='projects/mmdet3d_plugin/models/detectors/petr3d.py'
    def native(t):return [ast.dump(n) for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='self.lm_head.generate']
    proofs['Native_generate_AST']=native(old(p))==native(ast.parse(Path(p).read_text()))
    for p,h in before.items():
        if p.startswith(('worldstream/m1/','projects/mmdet3d_plugin/models/dense_heads/','projects/mmdet3d_plugin/models/backbones/')):
            proofs[p]=Path(p).is_file() and sha256(p)==h
    assert all(proofs.values()),[k for k,v in proofs.items() if not v]
    write(root/'source_audit.json',dict(status='PASS',compiled=compiled,broken_imports=broken,
        retired_files=removed,changed_files=changed,semantic_R0_parameters_identical=True,
        preserved_numerical_and_model_source=proofs,baseline_commit=baseline))
    print('R0_SOURCE_AUDIT_PASS',compiled,'files',len(removed),'retired')


def compare(root):
    from .frozen_main_report import rows
    before=list(rows(root/'before/paired_gpu_receipts.jsonl'))
    after=list(rows(root/'after/paired_gpu_receipts.jsonl'))
    keyed=lambda rs:{(r['pair_id'],r['frame_id']):r for r in rs}
    aa,bb=keyed(before),keyed(after)
    assert aa.keys()==bb.keys()
    mismatch=[];counts=Counter();workloads={}
    for key,a in aa.items():
        b=bb[key]
        for arm in ['A','C']:
            x,y=a[arm],b[arm];counts['model_calls_compared']+=1
            for field in ['input_digest','prompt_token_ids','questions','camera_ids','timestamp',
                          'request_seeds','feature_digest','visual_quality','m1_commit_count']:
                if x[field]!=y[field]:mismatch.append(dict(key=key,arm=arm,field=field))
            for qi,(q,r) in enumerate(zip(x['requests'],y['requests'])):
                counts['QA_compared']+=1
                for field in ['output_token_ids','llm_skipped','request_seed','sampling_first_forward_rng']:
                    if q.get(field)!=r.get(field):
                        mismatch.append(dict(key=key,arm=arm,question=qi,field=field,
                            before=q.get(field),after=r.get(field)))
                counts[arm+'_FAST_before']+=bool(q['llm_skipped'])
                counts[arm+'_FAST_after']+=bool(r['llm_skipped'])
            if x['answers']!=y['answers']:mismatch.append(dict(key=key,arm=arm,field='answers'))
        w=workloads.setdefault(a['workload_id'],dict(before_A_ms=0.,before_C_ms=0.,after_A_ms=0.,after_C_ms=0.,frames=0))
        w['frames']+=1
        for stage,row in [('before',a),('after',b)]:
            for arm in ['A','C']:w[stage+'_'+arm+'_ms']+=row[arm]['e2e_wall_ms']
    for w in workloads.values():
        for key in ['before_A_ms','before_C_ms','after_A_ms','after_C_ms']:w[key]/=w['frames']
        w['before_speedup']=w['before_A_ms']/w['before_C_ms']
        w['after_speedup']=w['after_A_ms']/w['after_C_ms']
        w['C_latency_change_fraction']=w['after_C_ms']/w['before_C_ms']-1
    result=dict(status='PASS' if not mismatch else 'REVIEW_DIFFERENCES',
        paired_frames=len(aa),counts=dict(counts),mismatches=mismatch,workloads=workloads,
        scope='Two complete exposed logs, all five loads, same input and seed; not all-18-log replay or independent TEST',
        timing='Actual synchronized frame E2E, observer overhead included. Wall times may vary; not bitwise timing equality.')
    write(root/'before_after_comparison.json',result)
    print(json.dumps(dict(status=result['status'],counts=result['counts'],mismatches=len(mismatch),workloads=workloads),indent=2))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['audit','prepare','compare'])
    p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args()
    {'audit':audit,'prepare':prepare,'compare':compare}[a.stage](a.root)


if __name__=='__main__':main()
