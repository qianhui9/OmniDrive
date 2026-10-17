"""Pre-execution identity/coverage audit, explicitly not measured GPU evidence."""
import hashlib,pickle
from collections import defaultdict
from pathlib import Path
from .identity import validate_sequence
from .audit import write


def main():
    root=Path('outputs/system_optimization_v2');path=root/'planned_sequence_protocol.json'
    if path.exists():raise ValueError('Preflight already frozen')
    source=Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    raw=pickle.load(source.open('rb'))['infos'];groups=defaultdict(list)
    for info in sorted(raw,key=lambda x:x['timestamp']):groups[info['scene_token']].append(info)
    plans=[]
    for name,start,count,frames,stride in [('smoke',100,2,2,1),('single',102,4,4,1),
            ('multi',102,4,2,1),('long_stream',106,4,32,1),('stride_refresh',110,2,4,2),('auto_validation',112,2,4,1)]:
        sequences=[]
        for scene,infos in sorted(groups.items())[start:start+count]:
            selected=infos[::stride][:frames]
            if len(selected)!=frames:raise ValueError('Insufficient continuous frames: '+scene)
            sequences.append(dict(scene=scene,**validate_sequence(selected,scene)))
        if len(sequences)!=count:raise ValueError('Insufficient independent scenes')
        plans.append(dict(run=name,scene_start=start,scenes=sequences,frame_stride=stride))
    write(path,dict(status='annotation_preflight_only_not_gpu_execution',plans=plans,
        dataset_index_semantics='timestamp sorted; execution must still validate actual Dataset triple',
        outcome_based_scene_replacement=False,quality_targets_sha256=hashlib.sha256((root/'quality_targets.json').read_bytes()).hexdigest()))
    print('Identity preflight passed for',sum(len(p['scenes']) for p in plans),'planned run-scenes')


if __name__=='__main__':main()
