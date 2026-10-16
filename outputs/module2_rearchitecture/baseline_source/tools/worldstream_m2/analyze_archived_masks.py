"""Reanalyse preserved 640x640 M1 masks; never merge archived timings with new runs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from worldstream.m1.mask_codec import decode_mask
from .refinement_common import ROOT
from .common import write


def main():
    source=Path('outputs/module2/integration_final/m1_m2_exact_1.jsonl');rows=[]
    for line in source.read_text().splitlines():
        item=json.loads(line)['precheck'];cameras={}
        for cid,record in item['invalidation_regions'].items():
            invalid=decode_mask(record['exact_invalidation_mask']);h,w=invalid.shape
            if (h,w)!=(640,640):raise ValueError('Archive is not model-input coordinates')
            stable=(~invalid).reshape(40,16,40,16).all(axis=(1,3));wins=[]
            for y in range(0,40,16):
                for x in range(0,40,16):
                    tile=stable[y:y+16,x:x+16]
                    wins.append(dict(row=y//16,col=x//16,stable=int(tile.sum()),valid_tokens=int(tile.size),candidate=bool(tile.all())))
            components,n=label(stable);counts=np.bincount(components.ravel())[1:]
            cameras[cid]=dict(stable_pixels=int((~invalid).sum()),stable_patches=int(stable.sum()),
                stable_patch_mask=stable.astype(int).tolist(),windows=wins,largest_component=int(counts.max()) if n else 0)
        rows.append(dict(frame_id=item['frame_id'],cameras=cameras))
    write(ROOT/'archived_m1_spatial_diagnosis.json',dict(source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_status='Archived original M2 real-model M1 masks, model-input 640x640. Reanalysis only; no archived timings merged.',
        unknown_changed_split_available=False,rows=rows))


if __name__=='__main__':main()
