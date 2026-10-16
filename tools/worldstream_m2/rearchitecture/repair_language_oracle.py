"""Re-execute original full baseline in a fresh directory after interrupted-run output collision."""
import copy
import json
from pathlib import Path
import random
from time import perf_counter
import numpy as np
import torch
from tools.worldstream_m2.refine_integration import planning_metric
from .common import ROOT,write,memory,quality_pass


@torch.no_grad()
def repair(model,dataset):
    from mmcv.parallel import scatter,collate
    spec=json.load(open(ROOT/'protocol.json'))['controlled_scenes'][0]
    folder=ROOT/'language_oracle_recheck'
    if folder.exists():raise ValueError('Fresh oracle directory required')
    folder.mkdir()
    model.test_flag=False;model.m1_manager=None;model.m2_runtime=None
    model.save_path=str(folder/'language')+'/'
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    records=[];base=None
    for ordinal,index in enumerate(spec['indices']):
        batch=scatter(collate([dataset[index]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
        image=batch['img'][0]
        if base is None:base=image.clone()
        image.copy_(base);k=ordinal%8
        if k==2:image[0,0,:,:16,:16]+=.03
        elif k==3:image[0,:3]+=.03
        elif k==4:image[0,:5]+=.03
        elif k==5:image+=.03
        elif k==6:image[0,0,:,:256,:256]+=.03
        elif k==7:image[0,:2,:,:256,:256]+=.03
        torch.cuda.synchronize();tick=perf_counter();out=model(return_loss=False,rescale=True,**batch)
        torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
        actual=memory(model)
        saved=np.load(ROOT/'integration_final'/f'mixed_baseline_{spec["scene"]}_{ordinal}_memory.npz')
        exact=all(np.array_equal(v,saved[f'{branch}.{name}'],equal_nan=True)
            for branch,values in actual.items() for name,v in values.items() if isinstance(v,np.ndarray))
        text=out[0].get('text_out');assert text,'All baseline QA must actually execute'
        records.append(dict(ordinal=ordinal,index=index,token=dataset.data_infos[index]['token'],
            native_memory_exact_to_original_baseline=exact,text=text,
            planning=planning_metric(text,dataset.data_infos[index]),complete_wall_ms=wall))
    if not all(r['native_memory_exact_to_original_baseline'] for r in records):
        write(ROOT/'language_oracle_recheck.json',dict(status='failed_memory_equivalence',rows=records))
        raise AssertionError('Fresh baseline Memory does not match; cannot repair language-only evidence')
    aggregate=ROOT/'latency_results.json';data=json.load(open(aggregate))
    write(ROOT/'pre_language_oracle_recheck.json',data)
    repaired=0
    for q in data['quality']:
        if q['case']!='mixed' or q['scene']!=spec['scene']:continue
        r=records[q['ordinal']]
        file=ROOT/'integration_final'/f'mixed_{q["mode"]}_0_{spec["scene"]}'/'language'/r['token']
        text=json.load(open(file));q['text_equal']=text==r['text']
        q['planning']=planning_metric(text,dataset.data_infos[q['index']])
        q['baseline_planning']=r['planning']
        q['planning_equal']=q['planning'].get('trajectory')==r['planning'].get('trajectory')
        q['frozen_targets_pass']=quality_pass(q)
        q['language_oracle']='fresh_complete_16_frame_baseline_recheck_native_memory_equal'
        repaired+=1
    for run in data['runs']:
        if run['case']=='mixed' and run['mode']=='baseline' and run['scene']==spec['scene']:
            for f in run['frames']:
                if f['ordinal']<4:f['timing_valid']=False;f['reason']='prior interrupted generation path existed; excluded from performance'
    write(aggregate,data)
    write(ROOT/'integration_final/runs_annotated.json',data['runs'])
    write(ROOT/'integration_final/quality_rechecked.json',data['quality'])
    write(ROOT/'correctness_results.json',dict(quality=[q for q in data['quality'] if q['case']=='mixed'],
        notes='Language-only oracle collision repaired by fresh full baseline; all 16 native memory states exactly reproduced. Earlier records retained in pre_language_oracle_recheck.json. Mixed single-run times are not performance claims.'))
    write(ROOT/'language_oracle_recheck.json',dict(status='completed',rows=records,repaired_quality_rows=repaired,
        initial_invalid_timing_frames=4,performance_policy='mixed runs are diagnosis only; real/repeat measured runs have distinct fresh paths'))
    print('Fresh original baseline recheck complete, native Memory exactly reproduced:',len(records),flush=True)
