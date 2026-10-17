"""Correct historical coverage without modifying any original artifacts or timings."""
import argparse
from collections import defaultdict
import hashlib,json,pickle
from pathlib import Path
import numpy as np
from .identity import token_index,resolve_sequence
from .audit import write


def main(old='outputs/system_optimization',dest='outputs/system_optimization_v2'):
    old=Path(old);dest=Path(dest)
    annotation=Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    raw=pickle.load(annotation.open('rb'))['infos'];by_token=token_index(raw)
    # This is used ONLY to diagnose the old index assumption, never to join.
    sorted_infos=sorted(raw,key=lambda x:x['timestamp'])
    coverage=[];audits=[]
    for protocol in sorted(old.glob('*/protocol.json')):
        status=json.loads((protocol.parent/'status.json').read_text())
        if status.get('status')!='completed':continue
        executed=defaultdict(list);row_path=protocol.parent/'frames.jsonl'
        if not row_path.exists():continue
        for line in row_path.open():
            r=json.loads(line);identity=r['frame']['identity']
            executed[identity['frame_id']].append(identity)
        for spec in json.loads(protocol.read_text())['scenes']:
            selected,corrected=resolve_sequence(spec,by_token,executed)
            legacy=[raw[i] for i in spec['indices']]
            idx_matches=[sorted_infos[i]['token']==t for i,t in zip(spec['indices'],spec['tokens'])]
            affected=[k for k in ('scene_name','description') if legacy[0].get(k)!=selected[0].get(k)]
            old_times=np.asarray([x['timestamp']/1e6 for x in legacy])
            old_pos=np.asarray([x['ego2global_translation'] for x in legacy])
            if any(x['token']!=t for x,t in zip(legacy,spec['tokens'])):
                affected+=['duration_s','measured_frame_gaps_s','ego_sampled_path_length_m','classes_observed','class_counts']
            coverage.append(dict(run=protocol.parent.name,scene=spec['scene'],split=spec['split'],**corrected,
                scope='Offline annotation coverage joined by token; never online admission'))
            audits.append(dict(run=protocol.parent.name,scene=spec['scene'],actual_execution_identity_valid=True,
                sorted_dataset_index_agrees=all(idx_matches),raw_index_agrees=all(x['token']==t for x,t in zip(legacy,spec['tokens'])),
                affected_fields=affected,legacy_duration_s=float(old_times[-1]-old_times[0]),
                corrected_duration_s=corrected['duration_s'],legacy_ego_path_m=float(np.linalg.norm(np.diff(old_pos,axis=0),axis=1).sum()),
                corrected_ego_path_m=corrected['ego_sampled_path_length_m'],
                protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                timing_invalidated=False))
    audit=dict(cause='NuScenesDataset.load_annotations sorts by timestamp; raw pickle array order differs',
        dataset_implementation='mmdetection3d/mmdet3d/datasets/nuscenes_dataset.py:load_annotations',
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),sequences=audits,
        old_results_modified=False,latency_records_preserved=True,
        conclusion='Coverage-only index mapping defect; all audited executed scene/token/timestamp identities are continuous and agree with raw token join')
    write(dest/'sequence_identity_audit.json',audit)
    write(dest/'corrected_sequence_coverage.json',dict(source=str(annotation),join='scene_token + sample_token + timestamp',sequences=coverage))
    text=['# 序列身份审计','','根因：Dataset 按时间全局排序，旧报告将 Dataset 下标用于未排序的原始 Pickle。',
        '按 token 连接并核对 scene/timestamp 后，实际执行连续；修正覆盖字段，不否定或改写原时延。','',
        '| Run | Scene | 原 duration s | 修正 duration s | 原 path m | 修正 path m |', '|---|---|---:|---:|---:|---:|']
    for a in audits:text.append(f"| {a['run']} | {a['scene']} | {a['legacy_duration_s']:.6f} | {a['corrected_duration_s']:.6f} | {a['legacy_ego_path_m']:.3f} | {a['corrected_ego_path_m']:.3f} |")
    (dest/'sequence_identity_audit.md').write_text('\n'.join(text)+'\n')
    print('Audited',len(audits),'sequences; affected',sum(bool(a['affected_fields']) for a in audits),'executed identity mismatches 0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',default='outputs/system_optimization');p.add_argument('--dest',default='outputs/system_optimization_v2')
    a=p.parse_args();main(a.old,a.dest)
