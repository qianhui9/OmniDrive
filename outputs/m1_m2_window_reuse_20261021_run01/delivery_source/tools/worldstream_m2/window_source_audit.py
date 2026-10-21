"""Audit actual published window origins and executor count conservation."""
import argparse
import json
import math
from pathlib import Path


def audit(directory, max_computation_age):
    root=Path(directory)
    runs=json.loads((root/'runs.json').read_text())
    failures=[];receipts=units=reused=0
    for run in runs:
        frame_ids=[f['frame'] for f in run['frames']]
        if len(frame_ids)!=len(set(frame_ids)):
            failures.append(dict(mode=run['mode'],scene=run['scene'],repeat=run['repeat'],reason='duplicate_frame_ids'))
        for ordinal,frame in enumerate(run['frames']):
            d=frame['runtime']
            if d.get('cache_boundary')!='window':continue
            receipts+=1
            location=dict(mode=run['mode'],scene=run['scene'],repeat=run['repeat'],ordinal=ordinal)
            def fail(reason):failures.append(dict(location,reason=reason))
            stages=d.get('stages',[])
            if sum(s.get('reused_windows',0) for s in stages)!=d['actual_skipped_block_computations']:
                fail('skipped_counter_mismatch')
            if sum(s.get('executed_windows',0) for s in stages)!=d['recomputed_windows']:
                fail('execution_counter_mismatch')
            context=d['cache_context'];now=context['timestamp']
            if context['scene_id']!=run['scene']:fail('scene_mismatch')
            for block,origins in d.get('output_window_provenance',{}).items():
                stage=next((s for s in stages if s.get('block')==int(block)),None)
                if stage is None or stage.get('total_windows')!=len(origins):fail('source_count_mismatch')
                if len(origins)%len(context['camera_ids']):fail('camera_partition_mismatch');continue
                count=len(origins)//len(context['camera_ids'])
                fresh_count=sum(s.get('computation_frame_id')==context['frame_id'] for s in origins)
                if stage is not None and (fresh_count!=stage.get('executed_windows') or
                        len(origins)-fresh_count!=stage.get('reused_windows')):
                    fail('execution_source_conservation_mismatch')
                for index,source in enumerate(origins):
                    units+=1
                    cid=context['camera_ids'][index//count]
                    if source.get('camera_id')!=cid or source.get('producer_layer')!=int(block):fail('source_position_mismatch')
                    timestamp=source.get('computation_timestamp')
                    if not isinstance(timestamp,(float,int)) or not math.isfinite(timestamp) or not 0<=now-timestamp<=max_computation_age:
                        fail('source_time_or_lease_mismatch')
                    fresh=source.get('computation_frame_id')==context['frame_id']
                    if not fresh:reused+=1
                    if fresh and timestamp!=now:fail('fresh_source_time_mismatch')
                    if not fresh and isinstance(timestamp,(float,int)) and timestamp>=now:fail('historical_source_not_older')
                    if fresh and d.get('cache_saved') and source.get('computation_version')!=d['cache_version']:
                        fail('fresh_source_version_mismatch')
                    version=source.get('computation_version')
                    if d.get('cache_saved') and (not isinstance(version,int) or not 0<version<=d['cache_version']):fail('source_version_mismatch')
                    if d.get('task_quality_certified') and not source.get('task_quality_certified',True):fail('quality_lineage_mismatch')
    return dict(scope='all recorded warmup and measured window receipts',receipts=receipts,
        checked_window_sources=units,historical_window_sources=reused,
        max_computation_age_s=max_computation_age,failures=failures,passed=not failures,
        limitation='Metadata/counter audit; full-task numerical quality is assessed separately in quality.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory')
    p.add_argument('--max-computation-age',type=float,default=1.)
    args=p.parse_args();result=audit(args.directory,args.max_computation_age)
    Path(args.directory,'source_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
