"""Actual M3 consumption at sparse checkpoints of full maintained streams.

The existing native M3 full path freezes precheck history before perception
commit. Offline state-only/augment requests consume that exact frozen object
and the real current carriers, without replaying or inventing temporal memory.
All timings belong to this supplemental workload, never single-service tables.
"""
import hashlib,inspect,json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import torch
from worldstream.m3 import M3Runtime
from .audit import write
from .experiment import seed
from .v2_probes import frozen_question


class CaptureFull(M3Runtime):
    def __init__(self):super().__init__(dict(mode='full',state_context='augment',offline_experiment=True))
    def generate(self,**kwargs):
        kwargs=dict(kwargs)
        if self.greedy:
            kwargs.update(do_sample=False);kwargs.pop('temperature',None);kwargs.pop('top_p',None)
        self.captured=kwargs
        result,diag=super().generate(**kwargs)
        self.answers=[x.detach().cpu() for x in result]
        return result,diag


@torch.no_grad()
def run(driver,source,dest):
    from mmcv.parallel import collate,scatter
    from tools.worldstream_m3.experiment import chosen_questions
    from tools.worldstream_m3.refinement import freeze_sample_questions
    source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    code=Path(__file__).read_bytes();(dest/'executed_probe.py').write_bytes(code)
    write(dest/'source_manifest.json',dict(sha256=hashlib.sha256(code).hexdigest()))
    protocol=json.loads((source/'protocol.json').read_text());original=json.loads((source/'questions.json').read_text())
    model,dataset=driver.model,driver.dataset;sink=SimpleNamespace(ids=[])
    native=inspect.getclosurevars(driver.native_generate).nonlocals['generate'];model.lm_head.generate=native
    for transform in dataset.pipeline.transforms:
        if hasattr(transform,'load_type'):transform.load_type=['planning','short','conv']
    checkpoints={7,15,31};frozen={}
    preflight=dest.parent/'long_state_question_preflight.json'
    predeclared={r['index']:r for r in json.loads(preflight.read_text())['checkpoints']} if preflight.exists() else {}
    # Freeze real extra questions before observing any candidate answer.
    for scene in protocol['scenes']:
        for ordinal,index in enumerate(scene['indices']):
            spec=dict(questions=list(original[str(index)]['questions']),ids=list(original[str(index)]['ids']))
            if ordinal in checkpoints:
                seed(index);sample=dataset[index];labels=sample['img_metas'][0].data['vlm_labels'].data
                ids=sample['input_ids'][0].data
                for i in chosen_questions(labels,5):
                    if labels[i] not in spec['questions'] and len(spec['questions'])<5:
                        spec['questions'].append(labels[i]);spec['ids'].append(ids[i].tolist())
                if len(spec['questions'])!=5:raise ValueError('Fewer than five real checkpoint questions')
                if index in predeclared:
                    expected=predeclared[index]
                    assert spec==dict(questions=expected['questions'],ids=expected['ids']),'Checkpoint questions differ from pre-execution CPU freeze'
            frozen[str(index)]=spec
    write(dest/'questions.json',frozen)
    write(dest/'protocol.json',dict(source=str(source),scene_tokens=[s['scene'] for s in protocol['scenes']],
        checkpoint_ordinals_zero_based=sorted(checkpoints),noncheckpoint_contract='Native single Planning Sampling',
        checkpoint_contract='Five actual questions, Greedy max_new_tokens=320, unchanged weights, same carriers and frozen M1 history',
        timing='Supplemental mixed-request stream and separate state candidate LM calls; excluded from ordinary E2E tables',
        independent_oracle_used_online=False))
    versions=[(n,p._version,p.dtype) for n,p in model.named_parameters()];records=[];stream=[]
    for scene in protocol['scenes']:
        driver.setup(model,'m1',sink,dest,512,10,False)
        supply=CaptureFull();model.m3_runtime=supply
        for ordinal,index in enumerate(scene['indices']):
            supply.greedy=ordinal in checkpoints
            seed(index);spec=frozen[str(index)]
            batch=scatter(collate([freeze_sample_questions(dataset[index],frozen_question(spec))],samples_per_gpu=1),[0])[0]
            seed(index+10000);model.save_path=str(dest/'stream'/str(index)/'language')+'/'
            torch.cuda.synchronize();start=perf_counter();model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
            frame=model.stream_runtime.last_record
            assert frame.identity.frame_id==dataset.data_infos[index]['token'] and frame.identity.scene_id==scene['scene']
            stream.append(dict(index=index,scene=scene['scene'],ordinal=ordinal,questions=len(spec['questions']),
                checkpoint=supply.greedy,inference_wall_ms=(perf_counter()-start)*1000,
                verified_anchor_count=frame.m1['valid_anchor_count'],m1=frame.m1))
            if not supply.greedy:continue
            captured=dict(supply.captured);state=captured['verified_state'];expected=supply.answers
            assert state.frame_id==frame.identity.frame_id and state.timestamp==frame.identity.timestamp
            assert state.state_epoch==model.m1_manager.store.generation and state.execution_signature==model.m1_manager.store.execution_signature
            assert state.diagnostics['source']=='M1 history before current perception commit'
            captured['world_state']=None  # candidates may read only the frozen precommit view
            for mode,context in (('state_only','only'),('state_full','augment')):
                runtime=M3Runtime(dict(mode='compact',offline_experiment=True,state_context=context))
                seed(index+10000);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=perf_counter()
                answers,diag=runtime.generate(**captured);torch.cuda.synchronize();elapsed=(perf_counter()-start)*1000
                equal=[torch.equal(a.cpu(),b) for a,b in zip(answers,expected)]
                record=dict(scene=scene['scene'],index=index,token=frame.identity.frame_id,ordinal=ordinal,mode=mode,
                    ego_to_world=driver.frame.ego_to_world.tolist(),timestamp=frame.identity.timestamp,
                    questions=spec['questions'],wall_ms=elapsed,gpu_peak_allocated=torch.cuda.max_memory_allocated(),
                    native_ids=[a.tolist() for a in expected],candidate_ids=[a.cpu().tolist() for a in answers],
                    native_text=[model.tokenizer.batch_decode(a,skip_special_tokens=True)[0] for a in expected],
                    candidate_text=[model.tokenizer.batch_decode(a,skip_special_tokens=True)[0] for a in answers],
                    complete_question_token_equality=equal,diagnostics=diag,verified_state_diagnostics=state.diagnostics,
                    frozen_historical_records=state.records)
                records.append(record);runtime.reset()
                with (dest/'queries.jsonl').open('a') as handle:handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            print('LONG STATE',scene['scene'],ordinal,'verified',len(state.records),flush=True)
        model.stream_runtime.close()
    assert versions==[(n,p._version,p.dtype) for n,p in model.named_parameters()]
    queries=[q['state_retrieval'] for r in records for q in r['diagnostics'].get('requests',[]) if q.get('state_retrieval')]
    hits=sum(q['admitted'] for q in queries)
    write(dest/'state_supply_results.json',dict(status='completed',unique_stream_frames=len(stream),independent_scenes=len(protocol['scenes']),
        checkpoint_frames=len({r['token'] for r in records}),actual_task_queries=len(queries),actual_state_input_activations=hits,
        task_query_hit_rate=hits/len(queries) if queries else None,all_complete_tokens_equal=all(all(r['complete_question_token_equality']) for r in records),
        no_ground_truth_used_online=True,weights_unchanged=True,production_approved=False,stream=stream,
        caveat='Mixed single-Sampling and checkpoint five-question-Greedy service. No aggregation into single-question speed or independent extra scenes.'))
