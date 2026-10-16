"""Offline-only necessity screen on actual preprocessed inputs; never skips M1 maintenance."""
from time import perf_counter
import numpy as np
from .common import write
from .refinement_common import ROOT,select_protocol


def main():
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py')
    cfg.data.test.test_mode=True;dataset=build_dataset(cfg.data.test)
    rows=[]
    for spec in select_protocol(dataset.data_infos):
        previous=None
        for index in spec['indices']:
            sample=dataset[index]
            image=sample['img'][0].data
            if hasattr(image,'detach'):image=image.detach().numpy()
            image=np.asarray(image)
            if image.ndim==5:image=image[0]
            if image.ndim!=4:raise ValueError(str(image.shape))
            if previous is not None:
                tick=perf_counter();rejected=[]
                for camera in range(6):
                    for y in range(0,640,256):
                        for x in range(0,640,256):
                            yy=np.array([y,(y+min(y+256,640)-1)//2,min(y+256,640)-1])
                            xx=np.array([x,(x+min(x+256,640)-1)//2,min(x+256,640)-1])
                            rejected.append(bool((image[camera][:,yy[:,None],xx[None,:]]!=
                                                  previous[camera][:,yy[:,None],xx[None,:]]).any()))
                ms=(perf_counter()-tick)*1000
                exact_windows=[]
                for camera in range(6):
                    for y in range(0,640,256):
                        for x in range(0,640,256):
                            sl=(camera,slice(None),slice(y,min(y+256,640)),slice(x,min(x+256,640)))
                            exact_windows.append(bool(np.array_equal(image[sl],previous[sl])))
                rows.append(dict(scene=spec['scene'],category=spec['category'],frame_id=dataset.data_infos[index]['token'],
                    sampled_proven_changed_windows=sum(rejected),total_windows=54,
                    necessary_condition_rules_out_all_reuse=all(rejected),
                    full_exact_equal_windows=sum(exact_windows),cpu_sample_screen_ms=ms))
            previous=image.copy()
    write(ROOT/'necessary_condition_diagnostic.json',dict(rows=rows,
        enabled_in_runtime=False,reason='Skipping M1 would require explicit history invalidation/reset and its own semantics/quality validation; this diagnostic does not bypass any precheck or commit.',
        guarantee='Changed sampled pixel only rejects equality; a matching sample never certifies a whole window.'))


if __name__=='__main__':main()
