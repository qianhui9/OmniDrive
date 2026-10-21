"""Frozen cached A/B/C paired replay; no detector/tracker inference or online qualification.

A uses the actual M1Config defaults and ordinary WorldStateManager. B/C use the
same config and transactions with the explicit offline-only research adapter.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import shutil
import sys
import numpy as np
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m1.state_types import json_value
from .dqtrack_three_way_policy import OfflineDQManager
from .dqtrack_policy_eval import evaluate_scene, rates, percentiles, scene_bootstrap

METHODS=('A','B','C')
RATIOS={'wrong_association_rate':('wrong','scored'),'effective_recall':('correct','eligible'),
    'association_coverage':('accepted','candidates'),'rejection_rate':('not_accepted','candidates'),
    'alignment_precision':('alignment_correct','alignment_scored'),
    'alignment_recall':('alignment_correct','alignment_possible'),
    'alignment_coverage':('aligned','tracks')}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(json_value(value),indent=2,allow_nan=False)+'\n')


def enrich(counts):
    result=dict(rates(counts));result['not_accepted']=counts['candidates']-counts['accepted']
    result['ratio_details']={k:dict(numerator=result.get(n,0),denominator=result.get(d,0),
        value=result.get(n,0)/result[d] if result.get(d,0) else None) for k,(n,d) in RATIOS.items()}
    result.update({k:v['value'] for k,v in result['ratio_details'].items()})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--cache-root',default='outputs/m1_dqtrack_policy_20261021')
    p.add_argument('--infos',default='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    p.add_argument('--data-root',default='data/nuscenes')
    p.add_argument('--test-log',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    root=Path(a.cache_root);protocol_path=root/'protocol.json'
    protocol=json.loads(protocol_path.read_text());assert sha(protocol_path)==(root/'protocol.sha256').read_text().strip()
    candidates_path=root/'omnidrive/model_observations.pkl'
    provenance=json.loads((root/'omnidrive/model_observation_provenance.json').read_text())
    assert sha(candidates_path)==provenance['prediction_sha256']
    assert sha(root/'omnidrive/data_selection.json')==provenance['selection_sha256']
    with open(a.infos,'rb') as f:infos={i['token']:i for i in pickle.load(f)['infos']}
    with candidates_path.open('rb') as f:observations=pickle.load(f)
    config=M1Config()  # actual defaults, no hidden history/threshold switches
    configs={m:dict(config=asdict(config),sha256=canonical(asdict(config)),strategy=m) for m in METHODS}
    dump(out/'config_manifest.json',dict(methods=configs,profile=dict(minimum_score=.5,
        consecutive_observations=3,freshness_s=1.,independently_calibrated=False),
        A='current actual defaults; no bridge calls or results reads',
        B='mature source-bound identity consistency and unqualified-conflict veto',
        C='B plus complete unique legal competing component identity mapping'))
    wanted=[];input_files={str(protocol_path),str(root/'protocol.sha256'),str(candidates_path),a.infos,
        str(root/'omnidrive/model_observation_provenance.json'),str(root/'omnidrive/data_selection.json'),
        str(root/'candidate_provenance_addendum.json'),a.test_log}
    checks=[];frames_manifest=[]
    for spec in protocol['development']:
        scene=spec['scene_id'];path=root/'dqtrack'/scene/'normalized_tracking.json'
        data=json.loads(path.read_text());manifest=data['m1_manifest'];mapping=manifest['frames']
        report_path=path.parent/'run_report.json';report=json.loads(report_path.read_text())
        assert report['actual_inference'] and report['run_id']==manifest['run_id']
        assert report['checkpoint_sha256']==manifest['checkpoint_sha256']
        assert set(mapping)==set(spec['samples'])==set(data['results'])
        assert manifest['coordinate_system']=='world' and manifest['size_order']=='wlh' and manifest['timestamp_unit']=='seconds'
        previous=None
        for token in spec['samples']:
            info=infos[token];ts=info['timestamp']/1e6;row=mapping[token]
            assert token in observations and observations[token].source in ('actual_perception','historical_full_model_detection_replay')
            assert info['scene_token']==scene==row['scene_id'] and row['sample_token']==token
            assert abs(row['timestamp']-ts)<=1e-6 and (previous is None or ts>previous)
            if previous is not None:assert ts-previous<=config.max_frame_gap_s
            previous=ts;wanted.append(info)
            frames_manifest.append(dict(scene_id=scene,frame_id=token,timestamp=ts,candidate_count=len(observations[token].anchors),source=observations[token].source))
            for camera in info['cams'].values():
                from .dataset import resolve_image
                image=resolve_image(camera['data_path'],a.data_root)
                assert image.is_file(),str(image)
                input_files.add(str(image))
        # Existing per-frame traces are evidence, never substituted for group A.
        old=root/'paired_owned'/scene/'geometry.json'
        old_data=json.loads(old.read_text());assert [t['frame_id'] for t in old_data['traces']]==spec['samples']
        input_files.update((str(path),str(report_path),str(old)))
        checks.append(dict(scene_id=scene,frames=len(spec['samples']),mapping_valid=True,existing_trace_valid=True))
    assert len(wanted)==161 and len({i['token'] for i in wanted})==161 and len(checks)==4
    for name in ('scene','sample','sample_annotation','instance','category','visibility'):
        file=Path(a.data_root)/'v1.0-trainval'/(name+'.json')
        if file.exists():input_files.add(str(file))
    sources={str(f):sha(f) for directory in ('worldstream/m1','tools/worldstream_m1','tests/worldstream_m1')
        for f in sorted(Path(directory).glob('*.py'))}
    inputs={path:dict(sha256=sha(path),bytes=Path(path).stat().st_size) for path in sorted(input_files)}
    manifest=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),sources=sources,inputs=inputs,
        input_bundle_sha256=canonical(inputs),frames=frames_manifest,command=['taskset','-c','0-3,6-31',sys.executable,*sys.argv],
        environment=dict(python=sys.version,cpu_affinity=sorted(os.sched_getaffinity(0))),
        protocol=dict(scenes=4,frames=161,oracle='class/size<=2/sensor support/<=1m/bidirectionally unique',
            recall_denominator='intersection of unique scored GT in adjacent frames',coverage_denominator='selected candidates after scene first frame',
            fragments='legacy count of gaps between scored GT observations, not official predicted FRAG',unknown='excluded from scored; never counted correct',
            execution='one replay per group/scene, cyclic group order by scene; no parameter search',timing='precheck+deepcopy+commit including bridge; excludes image load/GT; single replay timing is descriptive'),
        new_inference=False,GT_decision_input=False,production_hint_rejection_unchanged=True)
    dump(out/'source_manifest.json',manifest)
    for source in sources:
        target=out/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    (out/'source_changes.patch').write_text(subprocess.check_output(['git','diff'],text=True))
    shutil.copyfile(a.test_log,out/'tests.log')
    from .dataset import frame_from_info,load_oracle
    annotations=load_oracle(wanted,a.data_root)
    results={};all_traces={m:[] for m in METHODS};failures={m:dict(wrong=[],identity_switches=[],continuity_breaks=[],rejected=[]) for m in METHODS}
    with (out/'per_frame_trace.jsonl').open('x') as trace_file:
        for ordinal,spec in enumerate(protocol['development']):
            scene=spec['scene_id'];path=root/'dqtrack'/scene/'normalized_tracking.json'
            batches=[(frame_from_info(infos[t],a.data_root),observations[t]) for t in spec['samples']]
            results[scene]={};order=METHODS[ordinal%3:]+METHODS[:ordinal%3]
            for method in order:
                manager=WorldStateManager(config) if method=='A' else OfflineDQManager(config,method,path)
                result=evaluate_scene(batches,annotations,config,path,method,manager=manager)
                result['metrics']=enrich(result['metrics'])
                # No shared mutable outputs: evaluate_scene deep-copies every commit.
                for trace in result['traces']:
                    trace.update(method=method,input_bundle_sha256=manifest['input_bundle_sha256'])
                    trace_file.write(json.dumps(json_value(trace),allow_nan=False)+'\n');trace_file.flush()
                    failures[method]['wrong'] += [dict(frame_id=trace['frame_id'],scene_id=scene,**e) for e in trace['edges'] if e['status']=='wrong']
                    failures[method]['identity_switches'] += [dict(frame_id=trace['frame_id'],scene_id=scene,**e) for e in trace['identity_switch_events']]
                    if trace['continuity_break_instances']:failures[method]['continuity_breaks'].append(dict(frame_id=trace['frame_id'],scene_id=scene,instances=trace['continuity_break_instances']))
                    if trace['rejected_prediction_indices']:failures[method]['rejected'].append(dict(frame_id=trace['frame_id'],scene_id=scene,prediction_indices=trace['rejected_prediction_indices'],bridge_policy=(trace['dqtrack'] or {}).get('policy')))
                all_traces[method].extend(result['traces']);dump(out/(scene+'_'+method+'.json'),result)
                results[scene][method]={k:v for k,v in result.items() if k!='traces'}
                print(scene,method,json.dumps(result['metrics']),flush=True)
            del batches
    summary=dict(methods={},scenes=results,scope='EXPOSED_DEVELOPMENT_PARTIAL_ORACLE_NOT_OFFICIAL_TRACKING',production_default='off',source_manifest_sha256=sha(out/'source_manifest.json'))
    for method in METHODS:
        rows=[r[method] for r in results.values()]
        counts={k:sum(r['metrics'][k] for r in rows) for k,v in rows[0]['metrics'].items() if isinstance(v,int)}
        counts['unaligned_tracks']=sum((t['dqtrack'] or {}).get('unaligned_tracks',0) for t in all_traces[method])
        counts['resolved_edges']=sum(len((t['dqtrack'] or {}).get('policy',{}).get('resolved',[])) for t in all_traces[method])
        counts['vetoed_edges']=sum(len((t['dqtrack'] or {}).get('policy',{}).get('vetoes',[])) for t in all_traces[method])
        summary['methods'][method]=dict(metrics=enrich(counts),m1=percentiles([t['m1_ms'] for t in all_traces[method]]),
            bridge=percentiles([(t['dqtrack'] or {}).get('elapsed_ms',0) for t in all_traces[method]]),
            wrong_rate_scene_bootstrap=scene_bootstrap(rows),strata={})
        for stratum in set().union(*(r['strata'] for r in rows)):
            cc={}
            for r in rows:
                for k,v in r['strata'].get(stratum,{}).items():
                    if isinstance(v,int):cc[k]=cc.get(k,0)+v
            for k in ('accepted','candidates','correct','eligible'):cc.setdefault(k,0)
            # Legacy strata did not define candidate denominators: do not invent coverage.
            values=enrich(cc);values['ratio_details'].pop('association_coverage');values['ratio_details'].pop('rejection_rate')
            values.pop('association_coverage',None);values.pop('rejection_rate',None)
            summary['methods'][method]['strata'][stratum]=values
    dump(out/'summary.json',summary);dump(out/'failure_cases.json',failures)
    with (out/'comparison.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=['metric','A','B','C','B_minus_A','C_minus_A','B_relative_change','C_relative_change']);writer.writeheader()
        values={m:dict(summary['methods'][m]['metrics'],**{'m1_'+k:v for k,v in summary['methods'][m]['m1'].items()},**{'bridge_'+k:v for k,v in summary['methods'][m]['bridge'].items()}) for m in METHODS}
        for k,v in values['A'].items():
            if isinstance(v,(int,float)) or v is None:
                row=dict(metric=k,**{m:values[m][k] for m in METHODS})
                for m in ('B','C'):row[m+'_minus_A']=values[m][k]-v if v is not None and values[m][k] is not None else None;row[m+'_relative_change']=(values[m][k]-v)/v if v and values[m][k] is not None else None
                writer.writerow(row)
    with (out/'per_scene.csv').open('x') as f:
        keys=['scene_id','method','correct','wrong','scored','unknown','eligible','accepted','candidates','identity_switches','fragments','continuity_breaks','effective_recall','association_coverage','m1_p50_ms','m1_p95_ms']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for scene,rr in results.items():
            for method,r in rr.items():writer.writerow(dict(scene_id=scene,method=method,**{k:r['metrics'][k] for k in keys[2:-2]},m1_p50_ms=r['m1']['p50_ms'],m1_p95_ms=r['m1']['p95_ms']))
    unchanged=all(sha(path)==value['sha256'] for path,value in inputs.items()) and all(sha(path)==value for path,value in sources.items())
    assert unchanged
    checks_frames={m:[(t['scene_id'],t['frame_id'],t['timestamp'],t['selected_prediction_indices']) for t in ts] for m,ts in all_traces.items()}
    assert checks_frames['A']==checks_frames['B']==checks_frames['C']
    for m,ts in all_traces.items():
        for k,v in summary['methods'][m]['metrics'].items():
            if k in ts[0]['count_delta']:assert v==sum(t['count_delta'][k] for t in ts)
        assert all(len({i for i,j in t['pairs']})==len(t['pairs'])==len({j for i,j in t['pairs']}) for t in ts)
    assert all(t['dqtrack'] is None for t in all_traces['A'])
    dump(out/'verification.json',dict(input_checks=checks,frames_per_group={m:len(ts) for m,ts in all_traces.items()},
        input_and_sources_unchanged=unchanged,paired_selected_candidates_identical=True,trace_counts_recomputed=True,
        baseline_no_bridge=True,one_to_one=True,tests_log=a.test_log,tests_log_sha256=sha(a.test_log),
        tests=Path(a.test_log).read_text(),exceptions=[],new_model_inference=False,
        bridge_inference_cost_included=False,default_config_sha256=configs['A']['sha256']))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
