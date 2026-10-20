"""Native QA census timing and controlled fail-closed original-LLM requests."""
import copy
import hashlib
from pathlib import Path
from time import perf_counter
import torch
from worldstream.m3 import M3Runtime
from worldstream.m3.visual_state import base_model
from worldstream.m3.task_evidence import population_task
from tools.worldstream_m1.io_utils import write_json


@torch.no_grad()
def run_supplement(model,snapshots,dest):
    from mmcv import Config
    from mmcv.parallel import collate,scatter
    from mmdet3d.datasets import build_dataset
    from tools.worldstream_m3.task_serving_profile import InvocationTimer,stage
    dest=Path(dest);dest.mkdir(exist_ok=False)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py');cfg.data.test.test_mode=True
    dataset=build_dataset(cfg.data.test)
    for t in dataset.pipeline.transforms:
        if hasattr(t,'load_type'):t.load_type=['planning','short','conv']
    base=base_model(model.lm_head);device=base.get_input_embeddings().weight.device
    timer=InvocationTimer();timer.wrap(base,'forward','llm',stage);native=model.lm_head.generate;rows=[];queries=[];index=[0]
    def observe(*a,**k):
        timer.rows.clear();torch.cuda.synchronize();tick=perf_counter();out=native(*a,**k);torch.cuda.synchronize()
        q=queries[index[0]];rows.append(dict(question=q,task_type=population_task(q),source='actual_native_question',wall_ms=(perf_counter()-tick)*1000,stages=timer.collect(),prompt_tokens=int(k['inputs'].numel()),generated_tokens=int(out.shape[-1]-1),answer=model.tokenizer.batch_decode(out,skip_special_tokens=True)[0]))
        index[0]+=1;return out
    pts=model.simple_test_pts
    def entry(img_metas,**data):
        queries[:]=list(img_metas[0]['vlm_labels'].data);index[0]=0
        return pts(img_metas,**data)
    model.simple_test_pts=entry;model.lm_head.generate=observe
    scenes=set()
    try:
        for snap in snapshots:
            if snap['scene'] in scenes:continue
            scenes.add(snap['scene']);i=next(i for i,v in enumerate(dataset.data_infos) if v['token']==snap['frame'])
            model.test_flag=False;model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
            model.save_path=str(dest/'native_all'/snap['scene'])+'/'
            batch=scatter(collate([dataset[i]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
            first=len(rows);model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
            for r in rows[first:]:r.update(scene=snap['scene'],frame=snap['frame'])
            print('NATIVE_ALL',snap['scene'],len(rows)-first,flush=True)
    finally:model.simple_test_pts=pts;model.lm_head.generate=native
    write_json(dest/'native_all_questions.json',dict(population='exploratory_reused_population',rows=rows,selection='Every native planning/short/conversation/VQA request of the first frame in each of the same two service scenes; no answer-based selection',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    from projects.mmdet3d_plugin.datasets.utils.data_utils import preprocess
    snap=snapshots[0];vision=snap['carriers'].to(device)
    question='What is the distance to observed object fixture-1 in meters?'
    i=next(i for i,v in enumerate(dataset.data_infos) if v['token']==snap['frame']);location=dataset.data_infos[i]['location'].split('-')[0]
    ids=preprocess([[dict(**{'from':'human'},value='<image>\nYou are driving in '+location+'. '+question),dict(**{'from':'gpt'},value='')]],model.tokenizer,True,False)['input_ids'][0].to(device)
    context=dict(scene_id=snap['scene'],frame_id=snap['frame'],timestamp=1.,camera_ids=['six_real_cameras'],state_execution_signature='controlled-fixture',source_version='v1')
    record=dict(referent_id='fixture-1',scene_id=snap['scene'],frame_id=snap['frame'],timestamp=1.,execution_signature='controlled-fixture',source_frame=snap['frame'],source_version='v1',current_observation_available=True,current_fields=dict(center_ego_m=[3.,4.,0.]),contradiction_free=True,units='m',identity_evidence_origin='independent_current_observation',evidence_origin='controlled_functionality_fixture')
    negatives=[]
    for name,records in [('missing',[]),('ambiguous',[record,record]),('stale',[dict(record,timestamp=0.)]),('no_current',[dict(record,current_observation_available=False)]),('conflict',[dict(record,contradiction_free=False)]),('wrong_units',[dict(record,units='cm')]),('no_independent_certificate',[record])]:
        runtime=M3Runtime(dict(mode='exact',offline_experiment=True,task_evidence_enabled=True,experimental_direct_query=True,shadow_evaluation=False))
        timer.rows.clear();torch.cuda.synchronize();tick=perf_counter()
        out,diag=runtime.generate(model.lm_head,[ids],vision,context=context,questions=[question],task_evidence=records,tokenizer=model.tokenizer,do_sample=False,max_new_tokens=320,use_cache=True)
        torch.cuda.synchronize();negatives.append(dict(case=name,wall_ms=(perf_counter()-tick)*1000,diagnostics=diag,stages=timer.collect(),answer=model.tokenizer.batch_decode(out[0],skip_special_tokens=True)[0]))
        print('NEGATIVE',name,diag['actual_direct_requests'],flush=True)
    timer.restore();write_json(dest/'negative_execution.json',dict(population='controlled_functionality_only',records_are_physical_truth=False,quality_certificate_provided=False,rows=negatives))
