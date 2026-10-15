"""Counterfactual diagnostic measurements of gates masked by old short-circuit checks.

These checks never alter online decisions. Unknown prerequisites are kept as null.
"""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.geometry import project_anchor,intersect_area,region_area
from worldstream.m1.observation_validator import ObservationValidator
from .dataset import select_infos,frame_from_info,prediction_outputs
from .io_utils import write_json,read_prediction_subset


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/module1_refinement')
    root=Path(parser.parse_args().output);root.mkdir(parents=True,exist_ok=True)
    names=('replay_strict','replay_second_scene','replay_interval_1s')
    manifests={n:json.loads((Path('outputs/module1')/n/'manifest.json').read_text()) for n in names}
    selections={n:select_infos(m['args']['infos'],m['scene_id'],m['args']['frames'],m['args']['stride']) for n,m in manifests.items()}
    predictions=read_prediction_subset(manifests[names[0]]['args']['predictions'],{i['token'] for seq in selections.values() for i in seq})
    records=[];summary={}
    for name,infos in selections.items():
        config=M1Config.load(manifests[name]['config']);manager=WorldStateManager(config)
        validator=ObservationValidator(replace(config,require_visibility_evidence=False))
        triggers=Counter();sole=Counter();first=Counter();funnel=Counter();unmeasured=Counter();counts=0
        stages=['age','motion','calibration','observable_view','search_uncertainty','history','appearance','occlusion','visibility']
        for info in infos:
            frame=frame_from_info(info,'data/nuscenes');result=manager.precheck(frame)
            projections={aid:{cid:project_anchor(a,c,config) for cid,c in frame.cameras.items()} for aid,a in manager.store.anchors.items()}
            for aid,a in manager.store.anchors.items():
                views={}; flags={k:None for k in stages}
                for cid,cam in frame.cameras.items():
                    p=projections[aid][cid];old=a.observation_support.get(cid)
                    observable=bool(p.region and p.coverage>=config.min_coverage and region_area(p.region)>=config.min_region_area_px)
                    occluded=bool(p.region and any(other!=aid and qs[cid].region is not None and qs[cid].depth<p.depth-config.occlusion_depth_margin_m
                        and intersect_area(p.region,qs[cid].region)>config.occlusion_overlap_fraction*region_area(p.region) for other,qs in projections.items()))
                    # Force geometry-valid only inside this offline diagnostic, preserving radius cap.
                    v=validator.validate(old,cam,replace(p,valid=True),False) if observable and old is not None else None
                    views[cid]=dict(observable=observable,projection_reason=p.reason,projection_valid=p.valid,
                        history=old is not None,appearance=v.observation_consistent if v else None,
                        residual=v.residual if v else None,occluded=occluded,visibility_mask_present=cam.visibility_mask is not None)
                eligible=[v for v in views.values() if v['observable']]
                flags.update(age=0<=frame.timestamp-a.source_timestamp<=config.max_state_age_s,
                    motion=a.anchor_type!='object' or a.semantic_class in ('barrier','traffic_cone') or a.velocity_reliable,
                    calibration=all(v['projection_reason'] not in ('invalid_calibration','asynchronous_pose_unknown') for v in views.values()),
                    observable_view=bool(eligible))
                if eligible:
                    flags.update(search_uncertainty=all(v['projection_reason']!='uncertainty_exceeds_search' for v in eligible),
                        history=all(v['history'] for v in eligible),
                        appearance=all(v['appearance'] for v in eligible) if all(v['appearance'] is not None for v in eligible) else None,
                        occlusion=all(not v['occluded'] for v in eligible),visibility=all(v['visibility_mask_present'] for v in eligible))
                failures=[k for k,v in flags.items() if v is False];unknown=[k for k,v in flags.items() if v is None]
                triggers.update(failures);unmeasured.update(unknown)
                if len(failures)==1 and not unknown:sole.update(failures)
                first.update([next((k for k in stages if flags[k] is not True),'accepted')])
                survive=True
                for k in stages:
                    survive &= flags[k] is True
                    funnel[k]+=int(survive)
                records.append(dict(run=name,frame_id=frame.frame_id,anchor_id=aid,flags=flags,cameras=views,actual_accepted=aid in result.valid_anchor_ids))
                counts+=1
            manager.commit(frame,prediction_outputs(predictions[frame.frame_id],config),result)
        summary[name]=dict(trials=counts,triggered=dict(triggers),not_measurable=dict(unmeasured),sole_blocker=dict(sole),
            first_diagnostic_failure=dict(first),funnel=dict(funnel),stages=stages)
    write_json(root/'independent_constraint_records.json',records)
    write_json(root/'independent_constraint_summary.json',summary)
    lines=['# 逐约束独立测量补充','','用于诊断旧短路日志遮蔽的遮挡/外观条件；只在离线诊断中强制计算几何范围外的描述，绝不修改在线precheck判定。每个Anchor及每个Camera的原子条件保存在independent_constraint_records.json。',
           'None表示前提缺失而无法测量，不能算成观测失败。唯一阻塞只统计其余约束全部已测量且通过。投影缺失/出视野归入observable_view；每相机projection_reason保留更细原因。诊断顺序与旧实际首失败顺序分别报告，不能混用。','']
    for name,s in summary.items():
        lines+=['## '+name,'','| 约束 | 重叠触发 | 未可测 | 唯一阻塞 | 顺序首阻塞 | 累积剩余 |','|---|---:|---:|---:|---:|---:|']
        for k in s['stages']:lines.append('| '+k+' | '+' | '.join(str(s[key].get(k,0)) for key in ('triggered','not_measurable','sole_blocker','first_diagnostic_failure','funnel'))+' |')
        lines+=['']
    (root/'diagnosis_independent.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
