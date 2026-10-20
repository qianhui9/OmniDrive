"""Final-source full sensor/head/temporal/LLM parity and divergent-step RNG."""
import copy
import hashlib
import json
import random
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from tools.worldstream_m1.io_utils import write_json
from tools.worldstream_m2.rearchitecture.common import arrays,memory,compare,quality_pass
from tools.worldstream_m2.refine_integration import planning_metric
from .profile import ROOT,pin,load
from .service import CapturingRuntime


@torch.no_grad()
def main():
    dest=ROOT/'final_source_confirmation';hashes=pin(dest);model,dataset=load(dest)
    snapshots=torch.load(ROOT/'service/carrier_snapshots.pt',map_location='cpu')
    snap=next(s for s in snapshots if s['scene']=='a04daf2d0f194b2ab2ff2a47dfebc1d7' and s['ordinal']==0)
    index=next(i for i,v in enumerate(dataset.data_infos) if v['token']==snap['frame'])
    for transform in dataset.pipeline.transforms:
        if hasattr(transform,'load_type'):transform.load_type=['planning','short','conv']
    random.seed(0);sample=dataset[index]
    from mmcv.parallel import collate,scatter,DataContainer
    native_pts=model.simple_test_pts;features=[];generation=[]
    def pts(img_metas,**data):
        img_metas[0]['vlm_labels']=DataContainer(snap['questions'].copy(),cpu_only=True)
        data['input_ids'][0]=[x.cuda() for x in snap['ids']]
        return native_pts(img_metas,**data)
    model.simple_test_pts=pts
    native_extract=model.extract_img_feat
    def extract(*a,**kw):
        out=native_extract(*a,**kw);features.append(out.detach());return out
    model.extract_img_feat=extract
    native_generate=model.lm_head.generate
    def generate(*a,**kw):
        out=native_generate(*a,**kw);generation.append(out.cpu().tolist());return out
    model.lm_head.generate=generate
    rows=[];reference=None;ref_ids=None
    for mode in ('original','decode_graph','guarded_prefix'):
        model.test_flag=False;model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
        if mode!='original':
            model.m1_manager=WorldStateManager(M1Config.load('configs/worldstream_m1_engineering_frozen.yaml'))
            model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='auto',reuse_boundary='window',state_interface='source_linked',cache_capture='always',cost_table='outputs/module2_refinement/break_even_results.json',experimental_conditional_reuse=False))
            model.m3_runtime=CapturingRuntime(dict(mode='full' if mode=='decode_graph' else 'exact',offline_experiment=True,decode_cuda_graph=mode=='decode_graph',identical_prefill_only=mode=='guarded_prefix'))
        model.save_path=str(dest/'language'/mode)+'/'
        random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
        batch=scatter(collate([copy.deepcopy(sample)],samples_per_gpu=1),[torch.cuda.current_device()])[0]
        features.clear();generation.clear();torch.cuda.synchronize();tick=perf_counter()
        output=model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
        actual=dict(output=arrays(output),feature=arrays(features[-1]),memory=memory(model),text=output[0]['text_out'],planning=planning_metric(output[0]['text_out'],dataset.data_infos[index]))
        ids=model.m3_runtime.last_ids if model.m3_runtime else generation.copy()
        if mode=='original':reference=actual;ref_ids=ids;q=None
        else:
            q=compare(actual,reference);q.update(frozen_targets_pass=quality_pass(q),token_ids_equal=ids==ref_ids)
        torch.save(dict(output=actual['output'],memory=actual['memory'],text=actual['text'],planning=actual['planning'],ids=ids),dest/(mode+'.pt'))
        rows.append(dict(mode=mode,wall_ms=wall,quality=q,ids=ids,m3=copy.deepcopy(getattr(model,'m3_diagnostics',{})) if model.m3_runtime else {}))
        write_json(dest/'full_model.json',rows)
    from .sampling_prefix import run
    sampled=run(model,snapshots,ROOT,dest)
    write_json(dest/'status.json',dict(completed=True,full_model_forwards=len(rows),full_model_quality_pass=all(r['quality']['frozen_targets_pass'] and r['quality']['token_ids_equal'] for r in rows if r['quality']),
        sampled_failed_queries=sum(len(r['queries']) for r in sampled),sources_unchanged=all(hashlib.sha256(__import__('pathlib').Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())))


if __name__=='__main__':main()
