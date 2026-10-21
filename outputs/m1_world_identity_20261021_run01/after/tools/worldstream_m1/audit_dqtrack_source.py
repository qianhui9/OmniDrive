"""Compare trusted local official configs against extracted checkpoint metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy


PAIRS=(
 ('detr3d_r101_size_900_1600_dq.py','detr3d_res101_size_900_1600_finetune_amota_36_7.py'),
 ('uvtr_r101_size_900_1600_dq.py','uvtr_r101_size_900_1600_finetune_amota_39_6.py'),
 ('petrv2_vov99_fb3_320_800_dq.py','petrv2_vov99_fb3_320_800_finetune_amota_44_6.py'),
 ('bevstereo_r50_fb3_size_512_1408_dq.py','bevstereo_r50_fb3_size_512_1408_finetune_amota_36_7.py'),
 ('bevstereo_r101_fb3_size_512_1408_dq.py','bevstereo_r101_fb3_size_512_1408_finetune_amota_40_7.py'))


def differences(source,metadata,path='model'):
    if isinstance(source,dict) and isinstance(metadata,dict):
        result=[]
        for key in sorted(source.keys()|metadata.keys()):
            if key not in source or key not in metadata:
                result.append(dict(path=path+'.'+key,source=source.get(key),checkpoint=metadata.get(key),
                                   status='field_missing_in_one_config'))
            else:result.extend(differences(source[key],metadata[key],path+'.'+key))
        return result
    if source!=metadata:return [dict(path=path,source=source,checkpoint=metadata,status='different')]
    return []


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True)
    p.add_argument('--checkpoint-configs',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.source_root)
    report=dict(source_root=str(root.resolve()),strict_loading_required=True,configs=[])
    for new,old in PAIRS:
        source=root/'projects/configs/dqtrack'/new
        checkpoint=Path(a.checkpoint_configs)/old
        source_model=runpy.run_path(str(source))['model']
        old_model=runpy.run_path(str(checkpoint))['model']
        report['configs'].append(dict(source_config=str(source),checkpoint_config=str(checkpoint),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_model_type=source_model['type'],checkpoint_model_type=old_model['type'],
            differences=differences(source_model,old_model)))
    report['source_files']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.cpp','.cu','.h')}
    with open(a.output,'x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
