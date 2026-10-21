"""Hash the actual selected RGB/LiDAR, info, annotation and output sources."""
from pathlib import Path
import hashlib
import json
import pickle
from .dataset import resolve_image,frame_from_info
from .io_utils import write_json
from worldstream.m1.input_contract import frame_digest

ROOT=Path('outputs/module1_quality_coverage_refinement')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    selection=json.loads((ROOT/'data_selection.json').read_text());wanted={t for s in selection['development'] for t in s['samples']}
    info_path=Path('data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl')
    infos={i['token']:i for i in pickle.load(info_path.open('rb'))['infos'] if i['token'] in wanted}
    files={str(info_path):digest(info_path)};records=[]
    for token in sorted(wanted):
        info=infos[token];paths=[resolve_image(info['lidar_path'],'data/nuscenes')]
        paths += [resolve_image(c['data_path'],'data/nuscenes') for c in info['cams'].values()]
        raw=[]
        for path in paths:
            name=str(path);files.setdefault(name,digest(path));raw.append(name)
        frame=frame_from_info(info,'data/nuscenes')
        records.append(dict(sample_token=token,scene_token=info['scene_token'],timestamp=info['timestamp']/1e6,
            actual_runtime_frame_sha256=frame_digest(frame),source_files=raw))
    for name in ('sample.json','sample_annotation.json','category.json','instance.json','scene.json','log.json'):
        path=Path('data/nuscenes/v1.0-trainval')/name;files[str(path)]=digest(path)
    for directory in ('module1_evidence_optimization','module1_identity_surface_refinement'):
        for name in ('model_observations.pkl','model_observation_provenance.json'):
            path=Path('outputs')/directory/name;files[str(path)]=digest(path)
    write_json(ROOT/'data_manifest.json',dict(hash='SHA256',frames=records,files=files,
        input_scope='Real RGB, camera intrinsics and poses via exact serialized info and runtime frame digest; actual LiDAR and annotation tables are scoring-only',
        no_oracle_online=True))
    print('data source hashes',len(files),'runtime frames',len(records))


if __name__=='__main__':main()
