"""Hash actual selected sensor/QA inputs and metadata, without loading a model."""
from pathlib import Path
import json
import os
import pickle
from mmcv import Config
from tools.worldstream_m123.benchmark.common import write, sha256

root=Path(__file__).resolve().parent
assert not ({4,5} & set(os.sched_getaffinity(0)))
cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
manifest=[];metadata={};checked={}
for role,base,key in [('train',root,'train'),('val',root/'natural_audit','test')]:
    source=Path(cfg.data[key].ann_file)
    metadata[str(source)]=sha256(source)
    infos=pickle.load(source.open('rb'))['infos']
    lookup={r['token']:r for r in infos}
    sequence=json.loads((base/'bridge_sequence.json').read_text())
    for token in sequence['tokens']:
        info=lookup[token];cameras=[]
        for camera,record in info['cams'].items():
            p=Path(record['data_path'])
            if str(p) not in checked:checked[str(p)]=sha256(p)
            cameras.append(dict(camera_id=camera,path=str(p),sha256=checked[str(p)],timestamp=record['timestamp']))
        qa=Path('data/nuscenes/conv')/role/(token+'.json')
        manifest.append(dict(frame_id=token,scene_id=info['scene_token'],split=sequence['scene_splits'][info['scene_token']],
            log_id=sequence['scene_logs'][info['scene_token']],timestamp=info['timestamp'],
            cameras=cameras,original_qa_path=str(qa),original_qa_sha256=sha256(qa)))
for filename in ('scene.json','sample.json','sample_annotation.json','instance.json','category.json'):
    p=Path('data/nuscenes/v1.0-trainval')/filename
    metadata[str(p)]=sha256(p)
tokenizer={str(p):sha256(p) for pattern in ('tokenizer*','vocab*','special_tokens_map*','config.json')
    for p in Path('ckpts/pretrain_qformer').glob(pattern) if p.is_file()}
write(root/'real_data_sources.json',dict(frames=len(manifest),camera_images=len(checked),
    metadata_sha256=metadata,tokenizer_sha256=tokenizer,samples=manifest,
    GT_scope='Annotation tables used offline after bridge extraction only. Online generation uses no GT.'))
print('HASHED_REAL_SOURCES',len(manifest),len(checked))
