"""Actual-model negative control: EVA refresh versus reliable whole replay."""
import copy
from pathlib import Path
import random
import numpy as np
import torch
from tools.worldstream_m2.rearchitecture.common import arrays,memory,compare,quality_pass
from tools.worldstream_m1.io_utils import write_json
from tools.worldstream_m2.refine_integration import planning_metric


@torch.no_grad()
def recovery_probe(model,dataset,indices,dest):
    from mmcv.parallel import collate,scatter
    dest=Path(dest);dest.mkdir(exist_ok=False)
    for t in dataset.pipeline.transforms:
        if hasattr(t,'load_type'):t.load_type=['planning']
    samples={i:dataset[i] for i in indices};native=model.extract_img_feat;refs=[];rows=[]
    for case in ('full_reference','unqualified_prefix_input_then_full_EVA','legal_reset_and_reliable_replay'):
        model.test_flag=False;model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
        model.save_path=str(dest/case/'language')+'/'
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        captured=[];ordinal=[0]
        def extract(*a,**k):
            out=native(*a,**k)
            if case=='unqualified_prefix_input_then_full_EVA' and ordinal[0]==0:out=out+0.5
            captured.append(out.detach());return out
        model.extract_img_feat=extract
        for j,index in enumerate(indices):
            ordinal[0]=j;batch=scatter(collate([copy.deepcopy(samples[index])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
            out=model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
            actual=dict(output=arrays(out),feature=arrays(captured[-1]),memory=memory(model),text=out[0].get('text_out'),planning=planning_metric(out[0].get('text_out'),dataset.data_infos[index]))
            if case=='full_reference':refs.append(actual)
            else:
                q=compare(actual,refs[j]);q.update(case=case,frame=dataset.data_infos[index]['token'],ordinal=j,
                    full_EVA_this_request=not(case=='unqualified_prefix_input_then_full_EVA' and j==0),
                    frozen_targets_pass=quality_pass(q),independent_online_admission=False)
                rows.append(q)
        model.extract_img_feat=native
    write_json(dest/'results.json',dict(rows=rows,method='Controlled +0.5 EVA feature intervention on ordinal 0 only; original weights/inputs otherwise unchanged. Legal recovery discards downstream memory and replays all original frames from sequence start.',replay_full_quality_pass=all(r['frozen_targets_pass'] for r in rows if r['case']=='legal_reset_and_reliable_replay')))
