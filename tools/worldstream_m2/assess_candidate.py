"""Build evidence-bound M2 task assessments from actual paired full-model runs.

Limits are mandatory inputs fixed before evaluation. This tool never writes a
deployment table, changes thresholds, or treats feature finiteness as quality.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.worldstream_m123.benchmark.common import sha256, write
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality


def load(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def metric_at(data, path):
    for name in path.split('/'): data=data[name]
    if not finite(data): raise ValueError('Undefined/nonfinite metric: '+path)
    return data


def check_limits(rules):
    if rules.get('schema')!='m2_assessment_rules_v1': raise ValueError('Explicit assessment rules required')
    for key in ('min_frames','min_scenes'):
        if type(rules.get(key)) is not int or rules[key]<1: raise ValueError('Positive '+key+' required')
    if type(rules.get('min_cases_per_record',1)) is not int or rules.get('min_cases_per_record',1)<1:
        raise ValueError('Positive minimum per-record coverage required')
    for key in ('max_abs','mean_abs'):
        if not finite(rules.get('feature_error_limits',{}).get(key)) or rules['feature_error_limits'][key]<0:
            raise ValueError('Predeclared finite feature error limits required')
    for task in ('detection','map','vlm'):
        specs=rules.get('tasks',{}).get(task,[])
        if not specs: raise ValueError('Predeclared task metrics required: '+task)
        for spec in specs:
            if spec.get('direction') not in ('higher','lower') or not finite(spec.get('max_loss')) or spec['max_loss']<0:
                raise ValueError('Invalid metric direction/loss limit')
    if type(rules.get('max_critical_regression_requests')) is not int or rules['max_critical_regression_requests']<0:
        raise ValueError('Explicit critical regression limit required')


def run_data(folder):
    folder=Path(folder)
    status=load(folder/'status.json')
    if status.get('status')!='COMPLETED' or status.get('integrity')!='PASS':
        raise ValueError('Incomplete/invalid full-model run: '+str(folder))
    validation=load(folder/'checkpoint_validation.json')
    if validation.get('status')!='PASS': raise ValueError('Checkpoint was not strictly loaded')
    frames=rows(folder/'per_frame_metrics.jsonl');questions=rows(folder/'per_question_metrics.jsonl')
    tokens=load(folder/'sequence.json')['tokens']
    if [f['frame_id'] for f in frames]!=tokens or len(set(tokens))!=len(tokens):
        raise ValueError('Missing/duplicate/reordered actual frames')
    if not questions or any(not q.get('native_generation_executed') or q.get('reference') is None for q in questions):
        raise ValueError('M2 assessment requires labelled native-prefill VLM on every frame')
    if set(q['frame_id'] for q in questions)!=set(tokens): raise ValueError('Missing VLM frame coverage')
    keys=[(q['frame_id'],q['question']) for q in questions]
    if len(set(keys))!=len(keys): raise ValueError('Duplicate quality question')
    return dict(folder=folder,frames=frames,questions=questions,tokens=tokens)


def evaluation(folder,run):
    folder=Path(folder);summary=load(folder/'summary.json')
    if (summary.get('status')!='COMPLETED' or summary.get('sample_tokens')!=run['tokens']
            or summary.get('prediction_run')!=str(run['folder'].resolve())
            or summary.get('prediction_sha256')!=sha256(run['folder']/'predictions.pkl')):
        raise ValueError('Official evaluation is not bound to the actual predictions')
    for name in ('detection_metrics.json','map_metrics.json','scenes.json'):
        if summary.get('evidence_hashes',{}).get(name)!=sha256(folder/name):
            raise ValueError('Official metric evidence missing or modified: '+name)
    return summary


def compare_metrics(native,actual,specs):
    result=[]
    for spec in specs:
        a,b=metric_at(native,spec['metric']),metric_at(actual,spec['metric'])
        loss=(a-b) if spec['direction']=='higher' else (b-a)
        result.append(dict(spec,native=a,candidate=b,loss=loss,passed=loss<=spec['max_loss']))
    return result


def assess(candidate_path,baseline_dir,candidate_dir,baseline_eval,candidate_eval,rules_path,output,assessor):
    output=Path(output)
    if output.exists(): raise FileExistsError('Assessment output already exists')
    if not assessor.strip(): raise ValueError('Assessor/tool identity required')
    rules=load(rules_path);check_limits(rules)
    data=load(candidate_path)
    if not data.get('records'): raise ValueError('Candidate contains no measured partial windows')
    base,actual=run_data(baseline_dir),run_data(candidate_dir)
    for run in (base,actual):
        binding_path=run['folder']/'m2_assessment_rules_binding.json'
        if not binding_path.is_file() or load(binding_path).get('sha256')!=sha256(rules_path):
            raise ValueError('Assessment rules were not bound before both model runs')
    if base['folder'].resolve()==actual['folder'].resolve(): raise ValueError('Reference and candidate must be separate runs')
    if base['tokens']!=actual['tokens']: raise ValueError('Unpaired frame selection/order')
    base_args,actual_args=load(base['folder']/'invocation.json'),load(actual['folder']/'invocation.json')
    if base_args.get('offline_wsgrr') or actual_args.get('offline_wsgrr')!='D' or actual_args.get('condition')!='B2':
        raise ValueError('Expected native M2 reference and full-model offline D candidate')
    if any((f.get('m2') or {}).get('actual_skipped_windows',0) for f in base['frames']):
        raise ValueError('Reference must use full current visual computation')
    for f,g in zip(base['frames'],actual['frames']):
        if (f['scene_id'],f['timestamp'],f['seed'])!=(g['scene_id'],g['timestamp'],g['seed']):
            raise ValueError('Scene/time/RNG seed pairing mismatch')
        if f['identity']['checkpoint_sha256']!=g['identity']['checkpoint_sha256']:
            raise ValueError('Different actual checkpoints')
    frame_map={f['frame_id']:f for f in actual['frames']}
    qbase,qactual=base['questions'],actual['questions']
    if [(q['frame_id'],q['question']) for q in qbase]!=[(q['frame_id'],q['question']) for q in qactual]:
        raise ValueError('Unpaired questions/order')
    quality=[]
    for a,b in zip(qbase,qactual):
        if a['reference']!=b['reference']: raise ValueError('Different GT answers')
        wa,wb=a.get('input_witness',{}),b.get('input_witness',{})
        if not wa or not wb or wa['ids']!=wb['ids'] or wa['generation']!=wb['generation']:
            raise ValueError('Input IDs/generation configuration missing or different')
        quality.append(dict(frame_id=a['frame_id'],question=a['question'],
            native_answer=a['answer'],candidate_answer=b['answer'],reference=a['reference'],
            **paired_quality(b['answer'],a['answer'],a['reference'])))
    det_base,det_actual=evaluation(baseline_eval,base),evaluation(candidate_eval,actual)
    vlm_base=dict(token_f1=sum(q['baseline_f1'] for q in quality)/len(quality))
    vlm_actual=dict(token_f1=sum(q['f1'] for q in quality)/len(quality))
    task_checks={task:compare_metrics(a,b,rules['tasks'][task]) for task,a,b in (
        ('detection',det_base['detection'],det_actual['detection']),
        ('map',det_base['map'],det_actual['map']),('vlm',vlm_base,vlm_actual))}
    critical=sum(q['critical_regressions']>0 for q in quality)
    reasons=[]
    scenes=len({f['scene_id'] for f in actual['frames']})
    if len(actual['frames'])<rules['min_frames']: reasons.append('insufficient_frames')
    if scenes<rules['min_scenes']: reasons.append('insufficient_scenes')
    if critical>rules['max_critical_regression_requests']: reasons.append('new_critical_vlm_regressions')
    if any(not c['passed'] for v in task_checks.values() for c in v): reasons.append('task_quality_limit_exceeded')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'vlm_pairs.json',dict(unique_questions=len(quality),pairs=quality,
        critical_regression_requests=critical,scope='Token F1 / numeric and lexical relation proxies, not human semantic certification'))
    tasks={}
    for task in ('detection','map','vlm'):
        artifact=output/(task+'_assessment.json')
        write(artifact,dict(task=task,checks=task_checks[task],
            baseline_run=str(base['folder'].resolve()),candidate_run=str(actual['folder'].resolve()),
            baseline_prediction_sha256=sha256(base['folder']/'predictions.pkl'),
            candidate_prediction_sha256=sha256(actual['folder']/'predictions.pkl'),
            baseline_evaluation_sha256=sha256(Path(baseline_eval)/'summary.json'),
            candidate_evaluation_sha256=sha256(Path(candidate_eval)/'summary.json'),
            rules_path=str(Path(rules_path).resolve()),rules_sha256=sha256(rules_path),
            vlm_pairs_sha256=sha256(output/'vlm_pairs.json')))
        passed=all(c['passed'] for c in task_checks[task]) and not reasons
        tasks[task]=dict(status='passed' if passed else 'failed',evaluator='official_nuScenes/OpenLane' if task!='vlm' else 'paired_token_f1_and_critical_lexical_proxies',
            metrics={c['metric']:c for c in task_checks[task]},path=str(artifact.resolve()),sha256=sha256(artifact))
    validation_path=(Path(candidate_path).parent/data['records'][0]['numerical_evidence']['path']).resolve()
    if sha256(validation_path)!=data['validation_sha256']: raise ValueError('Numerical evidence changed')
    validation=load(validation_path)
    cases={c['case']:c for c in validation['natural']+validation['controlled_microbenchmarks']}
    records=[]
    for row in data['records']:
        errors=list(reasons);covered=[];key=row['key'];kh=hashlib.sha256(key.encode()).hexdigest()
        evidence_path=(Path(candidate_path).parent/row['numerical_evidence']['path']).resolve()
        if evidence_path!=validation_path or row['numerical_evidence'].get('sha256')!=data['validation_sha256']:
            errors.append('numerical_evidence_binding_mismatch')
        for name in row.get('evidence_cases',[]):
            case=cases.get(name,{});frame=frame_map.get(case.get('frame_id'))
            stage=next((s for s in (frame or {}).get('m2',{}).get('stages',[]) if s.get('block')==row['block']),{})
            if (frame is None or stage.get('executed_windows')!=row['updated']
                    or stage.get('cost_key_sha256')!=kh or not stage.get('reused_windows',0)):
                errors.append('missing_actual_case_signature_count:'+name);continue
            covered.append(name)
            for metric,limit in rules['feature_error_limits'].items():
                value=case.get('difference',{}).get(metric)
                if not finite(value) or value>limit: errors.append('feature_error_limit:'+name+':'+metric)
        if not covered or len(set(covered))!=len(row.get('evidence_cases',[])): errors.append('incomplete_case_coverage')
        case_frames={cases[name]['frame_id'] for name in covered}
        if len(case_frames)<rules.get('min_cases_per_record',1): errors.append('insufficient_per_record_cases')
        subset=[q for q in quality if q['frame_id'] in case_frames]
        record_tasks=dict(tasks)
        if subset:
            checks=compare_metrics(dict(token_f1=sum(q['baseline_f1'] for q in subset)/len(subset)),
                dict(token_f1=sum(q['f1'] for q in subset)/len(subset)),rules['tasks']['vlm'])
            if any(not c['passed'] for c in checks): errors.append('per_record_vlm_quality_limit_exceeded')
            artifact=output/('vlm_'+kh+'_'+str(row['updated'])+'.json')
            write(artifact,dict(checks=checks,frames=sorted(case_frames),questions=len(subset),
                scope='Actual frames executing this signature/count; full M2 policy quality, not isolated causal attribution',
                paired_outputs_sha256=sha256(output/'vlm_pairs.json')))
            record_tasks['vlm']=dict(tasks['vlm'],path=str(artifact.resolve()),sha256=sha256(artifact),
                metrics={c['metric']:c for c in checks},
                status='passed' if tasks['vlm']['status']=='passed' and all(c['passed'] for c in checks) else 'failed')
        records.append(dict(key=key,updated=row['updated'],block=row['block'],covered_cases=covered,
            feature_error_limits=rules['feature_error_limits'],tasks=record_tasks,
            status='passed' if not errors else 'failed',rejection_reasons=errors))
    result=dict(schema='m2_window_task_quality_v1',independent=True,
        status=('passed' if records and all(r['status']=='passed' for r in records) else
            'partial' if any(r['status']=='passed' for r in records) else 'failed'),
        assessor=assessor,candidate_sha256=sha256(candidate_path),records=records,
        rules_path=str(Path(rules_path).resolve()),rules_sha256=sha256(rules_path),
        baseline_run=str(base['folder'].resolve()),candidate_run=str(actual['folder'].resolve()),
        sample_frames=len(actual['frames']),sample_scenes=scenes,unique_questions=len(quality),
        independence_scope='Separate native and candidate full-model task executions; not a claim of scene-disjoint threshold calibration',
        production_approvals_added=0)
    write(output/'quality_report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','baseline','candidate-run','baseline-eval','candidate-eval','rules','output','assessor'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    existed=Path(args.output).exists()
    try:
        result=assess(args.candidate,args.baseline,args.candidate_run,args.baseline_eval,args.candidate_eval,args.rules,args.output,args.assessor)
    except Exception:
        import traceback
        if not existed and Path(args.output).exists():
            write(Path(args.output)/'failure.json',dict(status='FAILED',error=traceback.format_exc()))
        raise
    print(json.dumps(dict(status=result['status'],records=len(result['records']),output=args.output)))
    raise SystemExit(0 if result['status']=='passed' else 1)
