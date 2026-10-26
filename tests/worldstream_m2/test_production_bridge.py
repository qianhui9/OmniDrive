"""Synthetic, temporary evidence only. Never changes deployed cost tables."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tools.worldstream_m2.assess_candidate import assess,check_limits
from tools.worldstream_m2.promote_costs import check_candidate,promote_candidate
from tools.worldstream_m2.production_acceptance import validate_rows
from tools.worldstream_m123.benchmark.common import sha256
from .test_cost_promotion import evidence


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def assessment_fixture(tmp_path):
    rules=tmp_path/'rules.json'
    put(rules,dict(schema='m2_assessment_rules_v1',min_frames=2,min_scenes=2,
        feature_error_limits=dict(max_abs=.5,mean_abs=.02),max_critical_regression_requests=0,
        tasks=dict(detection=[dict(metric='mean_ap',direction='higher',max_loss=0.)],
            map=[dict(metric='DET_l',direction='higher',max_loss=0.)],
            vlm=[dict(metric='token_f1',direction='higher',max_loss=0.)])))
    key=json.dumps(dict(schema=3,block_index=0,backbone_version='b',numerical_policy='n'))
    kh=hashlib.sha256(key.encode()).hexdigest()
    validation=tmp_path/'validation.json'
    put(validation,dict(backbone_fingerprint='b',numerical_policy='n',
        natural=[dict(case='case',frame_id='f0',difference=dict(exact=True,max_abs=.1,mean_abs=.001),
            stages=[dict(block=0,updated_windows=1)])],controlled_microbenchmarks=[]))
    candidate=tmp_path/'candidate.json'
    put(candidate,dict(schema_version=3,backbone_fingerprint='b',production_ready=False,
        formal_quality_policy=dict(approved_counts=[]),
        records=[dict(key=key,block=0,updated=1,evidence_cases=['case'],numerical_verified=True,quality_certified=False,
        incremental_gpu_ms=1.,full_gpu_ms=2.,incremental_wall_ms=1.,full_wall_ms=2.,
        numerical_evidence=dict(path=validation.name,sha256=sha256(validation)))],validation_sha256=sha256(validation)))
    folders=[]
    for variant in ('native','candidate'):
        d=tmp_path/variant;d.mkdir();folders.append(d)
        (d/'predictions.pkl').write_bytes(('unit fixture '+variant).encode())
        put(d/'status.json',dict(status='COMPLETED',integrity='PASS'))
        put(d/'checkpoint_validation.json',dict(status='PASS'))
        put(d/'m2_assessment_rules_binding.json',dict(path=str(rules),sha256=sha256(rules)))
        put(d/'sequence.json',dict(tokens=['f0','f1']))
        put(d/'invocation.json',dict(condition='B2',offline_wsgrr='D' if variant=='candidate' else None))
        frames=[dict(frame_id=f'f{i}',scene_id=f's{i}',timestamp=i,seed=i,identity=dict(checkpoint_sha256='w'),
            m2=dict(actual_skipped_windows=1 if variant=='candidate' else 0,
                stages=[dict(block=0,executed_windows=1,cost_key_sha256=kh,reused_windows=1)])) for i in range(2)]
        (d/'per_frame_metrics.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
        qs=[dict(frame_id=f'f{i}',question='How many cars?',answer='1 car',reference='1 car',
            native_generation_executed=True,input_witness=dict(ids={'sha256':'ids'},generation={'temperature':.1},
                carriers={'sha256':variant})) for i in range(2)]
        (d/'per_question_metrics.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in qs))
        ev=d/'eval';ev.mkdir()
        for name in ('detection_metrics.json','map_metrics.json','scenes.json'):put(ev/name,{})
        put(ev/'summary.json',dict(status='COMPLETED',sample_tokens=['f0','f1'],prediction_run=str(d.resolve()),
            prediction_sha256=sha256(d/'predictions.pkl'),detection=dict(mean_ap=.5),map=dict(DET_l=.5),
            evidence_hashes={name:sha256(ev/name) for name in ('detection_metrics.json','map_metrics.json','scenes.json')}))
    return candidate,folders[0],folders[1],rules


def call_assess(tmp_path,fixture):
    candidate,native,actual,rules=fixture
    return assess(candidate,native,actual,native/'eval',actual/'eval',rules,tmp_path/'assessment','unit_test_only')


def test_assessment_pairs_real_records_allows_intended_carrier_changes(tmp_path):
    result=call_assess(tmp_path,assessment_fixture(tmp_path))
    assert result['status']=='passed' and result['records'][0]['covered_cases']==['case']
    assert result['production_approvals_added']==0
    assert result['unique_questions']==2


def test_generated_assessment_connects_to_unmodified_cost_parser(tmp_path):
    from worldstream.m2.cost_model import CostModel
    fixture=assessment_fixture(tmp_path)
    result=call_assess(tmp_path,fixture)
    key=result['records'][0]['key']
    checks=check_candidate(fixture[0],tmp_path/'assessment/quality_report.json',{key},[1])
    assert checks['status']=='PASS'
    table=tmp_path/'unit_only_approved.json'
    promote_candidate(fixture[0],tmp_path/'assessment/quality_report.json',{key},[1],'unit_test_only',table)
    assert CostModel(str(table)).beneficial(key,1)==(True,None)


@pytest.mark.parametrize('fault',['predictions','metrics','ids','order','checkpoint','missing_gt','duplicate','status','rules_binding'])
def test_assessment_rejects_unbound_or_incomplete_inputs(tmp_path,fault):
    fixture=assessment_fixture(tmp_path);candidate,native,actual,rules=fixture
    if fault=='predictions':(actual/'predictions.pkl').write_bytes(b'modified')
    if fault=='metrics':put(actual/'eval/map_metrics.json',{'modified':True})
    if fault in ('ids','missing_gt','duplicate'):
        qs=[json.loads(s) for s in (actual/'per_question_metrics.jsonl').read_text().splitlines()]
        if fault=='ids':qs[0]['input_witness']['ids']={'sha256':'changed'}
        if fault=='missing_gt':qs[0]['reference']=None
        if fault=='duplicate':qs[1]=qs[0]
        (actual/'per_question_metrics.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in qs))
    if fault=='order':put(actual/'sequence.json',dict(tokens=['f1','f0']))
    if fault=='checkpoint':put(actual/'checkpoint_validation.json',dict(status='FAIL'))
    if fault=='status':put(actual/'status.json',dict(status='FAILED'))
    if fault=='rules_binding':put(actual/'m2_assessment_rules_binding.json',dict(sha256='changed'))
    with pytest.raises(ValueError):call_assess(tmp_path,fixture)


@pytest.mark.parametrize('fault',['quality','critical','count','key','feature','samples'])
def test_assessment_records_failed_evidence_without_approving(tmp_path,fault):
    fixture=assessment_fixture(tmp_path);candidate,native,actual,rules=fixture
    if fault in ('quality','critical'):
        qs=[json.loads(s) for s in (actual/'per_question_metrics.jsonl').read_text().splitlines()]
        qs[0]['answer']='2 cars'
        (actual/'per_question_metrics.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in qs))
    if fault in ('count','key'):
        fs=[json.loads(s) for s in (actual/'per_frame_metrics.jsonl').read_text().splitlines()]
        fs[0]['m2']['stages'][0]['executed_windows' if fault=='count' else 'cost_key_sha256']=2 if fault=='count' else 'wrong'
        (actual/'per_frame_metrics.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in fs))
    if fault in ('feature','samples'):
        r=json.loads(rules.read_text())
        if fault=='feature':r['feature_error_limits']['max_abs']=.01
        else:r['min_scenes']=3
        put(rules,r)
        for folder in (native,actual):put(folder/'m2_assessment_rules_binding.json',dict(sha256=sha256(rules)))
    result=call_assess(tmp_path,fixture)
    assert result['status']=='failed' and result['records'][0]['rejection_reasons']
    assert result['production_approvals_added']==0


def test_assessment_requires_predeclared_feature_limits(tmp_path):
    *_,rules=assessment_fixture(tmp_path);r=json.loads(rules.read_text())
    r['feature_error_limits']['max_abs']=None
    with pytest.raises(ValueError,match='Predeclared'):check_limits(r)


def test_promotion_dry_run_does_not_publish_and_reports_missing_count(tmp_path,eva):
    candidate,quality,keys,row=evidence(tmp_path,eva)
    before=set(tmp_path.iterdir())
    result=check_candidate(candidate,quality,keys,[0,99])
    assert result['status']=='FAIL' and result['counts'][0]['status']=='PASS'
    assert result['counts'][1]['status']=='FAIL' and set(tmp_path.iterdir())==before


def test_failed_record_cannot_hide_in_partial_report(tmp_path,eva):
    candidate,quality,keys,row=evidence(tmp_path,eva)
    q=json.loads(quality.read_text());q['status']='partial';q['records'][0]['status']='failed';put(quality,q)
    with pytest.raises(ValueError,match='did not pass'):
        promote_candidate(candidate,quality,keys,[0],'unit_fixture',tmp_path/'approved.json')


def test_actual_formal_direct_publication_then_nonzero_reuse(tmp_path,eva,context):
    from worldstream.m2 import IncrementalVisionRuntime
    from worldstream.m2.cost_model import cost_key,CostModel
    from worldstream.m2.dependency_planner import block0_input
    from worldstream.m1 import M1Config,WorldStateManager
    from .conftest import synthetic_approval
    from .test_reuse_diagnostics import step
    rt=IncrementalVisionRuntime(eva)
    image=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    records=[]
    with torch.no_grad():
        x=block0_input(eva,image.float()/255)
        for i in rt.prefix:
            key=cost_key(eva.blocks[i],x,i,model=eva,refresh_policy=rt.config.refresh_signature())
            # Tiny fixture geometry: two cameras, budget three windows each.
            records.append(dict(key=key,block=i,updated=6,numerical_verified=True,quality_certified=True,
                incremental_gpu_ms=.1,full_gpu_ms=1.,incremental_wall_ms=.1,full_wall_ms=1.,
                approval=synthetic_approval(tmp_path,key,6)))
            x=eva.blocks[i](x)
    table=tmp_path/'unit_only_formal.json'
    put(table,dict(schema_version=3,records=records,production_ready=True,formal_quality_policy=dict(approved_counts=[6])))
    manager=WorldStateManager(M1Config.load('configs/worldstream_m1_final.yaml'))
    with torch.no_grad():
        _,first,_=step(rt,manager,image,context,0,offline=False)
        _,second,_=step(rt,manager,image,context,1,offline=False)
    assert first['cache_saved'] and not hasattr(rt,'costs')
    assert second['actual_skipped_windows']>0 and not second.get('offline_experiment',False)
    result=validate_rows([dict(scene_id='s',m2=first),dict(scene_id='s',m2=second)])
    assert result['status']=='PASS' and result['window_reuse_ratio']>0
    broken=copy.deepcopy(second);broken['stages'][0]['planned_window_ids']=[]
    assert validate_rows([dict(scene_id='s',m2=first),dict(scene_id='s',m2=broken)])['status']=='FAIL'


def test_full_model_profiler_restores_actual_cache_and_m1_transaction(tmp_path,eva,context,monkeypatch):
    from worldstream.m2 import IncrementalVisionRuntime
    from worldstream.m2.cost_model import cost_key,CostModel
    from worldstream.m2.dependency_planner import block0_input
    from worldstream.m1 import M1Config,WorldStateManager
    from .test_reuse_diagnostics import step
    from tools.worldstream_m2.full_model_evidence import FullModelEvidence
    import tools.worldstream_m2.full_model_evidence as module
    def cpu_pairs(a,b,repeats,warmup):
        return [a(),b()],dict(wall_ms=1.,gpu_span_ms=1.),dict(wall_ms=2.,gpu_span_ms=2.)
    monkeypatch.setattr(module,'paired_measure',cpu_pairs)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:None)
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state:None)
    rt=IncrementalVisionRuntime(eva)
    image=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    records=[]
    with torch.no_grad():
        x=block0_input(eva,image.float()/255)
        for i in rt.prefix:
            key=cost_key(eva.blocks[i],x,i,model=eva,refresh_policy=rt.config.refresh_signature())
            records.append(dict(key=key,block=i,updated=6,numerical_verified=False,quality_certified=False,
                incremental_gpu_ms=1.,full_gpu_ms=2.,incremental_wall_ms=1.,full_wall_ms=2.))
            x=eva.blocks[i](x)
    table=tmp_path/'candidate_table.json';put(table,dict(schema_version=3,records=records))
    observer=FullModelEvidence(rt)
    manager=WorldStateManager(M1Config.load('configs/worldstream_m1_final.yaml'))
    with torch.no_grad():
        for i in range(2):
            _,diag,_=step(rt,manager,image,context,i,offline=True)
            saved=rt.cache;version=rt.cache.version;commits=manager.store.commit_count
            observer.collect(str(i))
            assert rt.cache is saved and rt.cache.version==version
            assert manager.store.commit_count==commits
    observer.finish(tmp_path,'synthetic_checkpoint')
    candidate=json.loads((tmp_path/'m2_window_costs_provisional.json').read_text())
    assert len(candidate['records'])==2 and not candidate['production_ready']
    assert all(r['evidence_cases']==['full_model:1'] for r in candidate['records'])


def test_worker_cost_overrides_are_explicit_and_isolated(tmp_path):
    from mmcv import Config
    from tools.worldstream_m123.benchmark.worker import effective_config,configure_m2_evidence
    from worldstream.m2 import M2Config
    path=tmp_path/'provisional.json'
    key=json.dumps(dict(schema=3,block_index=0,refresh_policy=M2Config().refresh_signature()))
    put(path,dict(schema_version=3,records=[dict(key=key,block=0,updated=6,
        incremental_gpu_ms=1.,full_gpu_ms=2.,incremental_wall_ms=1.,full_wall_ms=2.)]))
    args=NS(condition='B2',workload='language',offline_wsgrr='D',perception_only=False,
        m2_candidate_cost_table=str(path),m2_cost_table=None,m2_profile_evidence=True,window_budget=3)
    cfg=effective_config('B2','language',tmp_path)
    configure_m2_evidence(cfg,args,tmp_path)
    assert 'cost_table' not in cfg.model.m2_config
    assert 'cost_table' not in Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py').model.m2_config
    args.m2_candidate_cost_table=None;args.m2_profile_evidence=False;args.m2_cost_table=str(path);args.offline_wsgrr=None
    configure_m2_evidence(cfg,args,tmp_path)
    assert 'cost_table' not in cfg.model.m2_config
