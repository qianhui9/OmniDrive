"""Audit saved first-round decisions, then replay geometry for offline motion diagnosis."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from worldstream.m1 import M1Config, WorldStateManager, WorldAnchor
from worldstream.m1.geometry import predict_geometry, project_anchor
from .dataset import select_infos, frame_from_info, load_oracle, prediction_outputs
from .evaluate import oracle_assignment
from .io_utils import write_json, read_prediction_subset

ROOT = Path('outputs/module1_refinement')
RUNS = ('replay_strict', 'replay_appearance_ablation', 'replay_second_scene', 'replay_interval_1s')


def saved_funnel(path):
    stages = ['within_horizon', 'motion_known', 'calibration', 'reliable_view',
              'historical_description', 'observation', 'visibility']
    rows, reasons, view_reasons, resets = [], Counter(), Counter(), Counter()
    for line in (path/'state.jsonl').read_text().splitlines():
        d = json.loads(line)['precheck']['diagnostics']
        resets[str(d['reset_reason'])] += 1
        for aid, a in d['anchors'].items():
            views = list(a['cameras'].values()); eligible = [v for v in views if v['geometry_valid']]
            flags = [a['within_horizon'], a['motion_known'],
                     'calibration_or_camera_timing_unknown' not in a['reasons'], bool(eligible),
                     bool(eligible) and all(v['reason'] != 'no_historical_observation' for v in eligible),
                     bool(eligible) and all(v['observation_consistent'] for v in eligible),
                     bool(eligible) and all(v['visibility_and_coverage_valid'] for v in eligible)]
            rows.append(flags); reasons.update(set(a['reasons']))
            view_reasons.update(v['reason'] for v in views)
    x = np.array(rows, bool)
    funnel = [{'stage': 'all', 'remaining': len(rows)}]
    first = Counter()
    for row in rows:
        first[next((stages[i] for i, ok in enumerate(row) if not ok), 'accepted')] += 1
    for i, name in enumerate(stages):
        funnel.append({'stage': name, 'remaining': int(x[:, :i+1].all(1).sum()),
                       'failure_count_including_overlaps': int((~x[:, i]).sum()),
                       'sole_blocker_count': int((~x[:, i] & np.delete(x, i, 1).all(1)).sum()),
                       'first_failure': first[name]})
    return dict(funnel=funnel, anchor_reason_counts=dict(reasons), camera_reason_counts=dict(view_reasons),
                reset_counts=dict(resets), caveat='Saved short-circuit checks: unmeasured appearance is not an independent failure. '
                'Sole-blocker is conjunction ablation, not a causal estimate. Projection failures occur per camera.')


def main():
    global ROOT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT))
    ROOT = Path(parser.parse_args().output)
    ROOT.mkdir(exist_ok=True, parents=True)
    funnels = {r: saved_funnel(Path('outputs/module1')/r) for r in RUNS}
    write_json(ROOT/'validity_funnel.json', funnels)
    manifests = {r: json.loads((Path('outputs/module1')/r/'manifest.json').read_text()) for r in RUNS if r != 'replay_appearance_ablation'}
    selections = {r: select_infos(m['args']['infos'], m['scene_id'], m['args']['frames'], m['args']['stride']) for r, m in manifests.items()}
    all_infos = [i for v in selections.values() for i in v]
    oracle = load_oracle(all_infos, 'data/nuscenes')
    predictions = read_prediction_subset(next(iter(manifests.values()))['args']['predictions'], {i['token'] for i in all_infos})
    records = []
    for run, infos in selections.items():
        cfg = M1Config.load(manifests[run]['config']); manager = WorldStateManager(cfg)
        identities, source_anns = {}, {}
        for info in infos:
            frame = frame_from_info(info, 'data/nuscenes'); result = manager.precheck(frame)
            anns = oracle[frame.frame_id]; by_id = {a['instance_id']: a for a in anns}
            for aid, check in result.diagnostics['anchors'].items():
                a = manager.store.anchors[aid]; identity = identities.get(aid); cur = by_id.get(identity)
                prev = source_anns.get(aid)
                error = np.linalg.norm(predict_geometry(a, frame.timestamp).center-cur['geometry'].center) if cur else None
                displacement = np.linalg.norm(cur['geometry'].center-prev['geometry'].center) if cur and prev else None
                radii = []
                for cam in frame.cameras.values():
                    p = project_anchor(a, cam, cfg)
                    if p.region is not None and p.coverage >= cfg.min_coverage and p.depth > 0:
                        u = a.geometry_uncertainty + (cfg.velocity_uncertainty_mps*max(0, cam.timestamp-a.source_timestamp)
                            if a.semantic_class not in ('barrier','traffic_cone') else 0)
                        radii.append(float(np.ceil(cfg.uncertainty_px+cam.intrinsic[0,0]*u/p.depth)))
                records.append(dict(run=run, frame_id=frame.frame_id, anchor_id=aid, semantic_class=a.semantic_class,
                    age_s=check['age_s'], previous_identity_known=identity is not None, same_instance_present=cur is not None,
                    current_sensor_support=cur['has_sensor_support'] if cur else None,
                    current_visibility=cur['visibility'] if cur else None, center_error_m=error,
                    gt_displacement_m=displacement, gt_speed_mps=displacement/check['age_s'] if displacement is not None else None,
                    required_search_radius_px=radii, configured_radius_px=cfg.search_radius_px,
                    prediction_velocity_mps=float(np.linalg.norm(a.velocity_world)) if a.velocity_world is not None else None))
            outputs = prediction_outputs(predictions[frame.frame_id], cfg)
            current = [WorldAnchor(str(j), frame.scene_id, p.anchor_type, p.semantic_class, p.geometry_world, frame.timestamp) for j,p in enumerate(outputs.anchors)]
            assignments = oracle_assignment(current, anns)
            manager.commit(frame, outputs, result)
            indices = manager.last_commit_diagnostics.get('selected_prediction_indices', range(len(current)))
            for j, aid in zip(indices, manager.last_commit_diagnostics['committed_anchor_ids']):
                identities[aid] = assignments.get(j); source_anns[aid] = by_id.get(assignments.get(j))
        print('diagnosed', run, flush=True)
    write_json(ROOT/'motion_diagnosis.json', records)
    lines = ['# 第一轮失效诊断（算法修改前生成）', '', '依据：第一轮原始逐 Anchor/Camera 日志，以及相同预测、真实图像与离线 GT 的重新回放。GT 只在 precheck 后用于诊断。', '']
    for run in RUNS:
        lines += ['## '+run, '', '| 按现有判定顺序的约束 | 剩余 | 重叠失败 | 唯一阻塞 | 首失败 |', '|---|---:|---:|---:|---:|']
        for f in funnels[run]['funnel']:
            lines.append('| '+f['stage']+' | '+' | '.join(str(f.get(k,'—')) for k in ['remaining','failure_count_including_overlaps','sole_blocker_count','first_failure'])+' |')
        lines += ['', 'Anchor 原因：`'+json.dumps(funnels[run]['anchor_reason_counts'])+'`。', 'Camera 原因：`'+json.dumps(funnels[run]['camera_reason_counts'])+'`。', '']
        rs = [r for r in records if r['run']==run and r['previous_identity_known']]
        if rs:
            stages = [('历史身份已知', lambda r: True), ('同一实例仍存在',lambda r:r['same_instance_present']),
                      ('传感器支持',lambda r:r['current_sensor_support']), ('Visibility=4',lambda r:str(r['current_visibility'])=='4'),
                      ('中心误差≤1m',lambda r:r['center_error_m'] is not None and r['center_error_m']<=1)]
            subset=rs
            for name,pred in stages:
                subset=[r for r in subset if pred(r)]; lines.append('- Physical Oracle '+name+'：'+str(len(subset)))
            for key in ['gt_displacement_m','gt_speed_mps','center_error_m']:
                xs=[r[key] for r in rs if r[key] is not None]
                lines.append('- '+key+' P50/P95：'+str(np.percentile(xs,[50,95]).round(3).tolist() if xs else None))
            radii=[v for r in rs for v in r['required_search_radius_px']]
            lines.append('- 投影所需半径 P50/P95 px：'+str(np.percentile(radii,[50,95]).round(2).tolist() if radii else None))
            lines.append('')
    lines += ['## 根因及统计边界', '',
      '1. 所有回放 CameraFrame 都未提供 visibility_mask/geometry_support_mask。旧严格认证没有可用可见性证据；取消该约束的消融不能成为正式方案。',
      '2. 动态类别没有可靠速度，因此 motion_known=false；即使外观相似也不会被正式接受。0.5 秒运动误差和静态预测误差已单独保存。搜索半径由速度不确定性推导，常超过4px；不能仅扩大搜索就声称运动可靠。',
      '3. 旧 novelty 将未提供支持的像素全部置为 Unknown；所有回放的 Verified Stable Pixels=0、Unknown/Unsupported=100%。旧 changed_regions 将 inf（包括未测量）也算变化，不能据此报告 Observed Changed Pixels；该指标不可辨识，须修正。',
      '4. 周期刷新和 reset 只增加更新需求，不能解释其余帧仍然全图失效。新实现必须保留局部证据，即使其他区域未知。',
      '5. 多约束统计依赖现有短路日志；sole_blocker 是移除单一布尔约束的结果，不能把未执行的图像验证当作独立观测。完整逐相机原因及离线运动记录见同目录 JSON。',
      '', '下一步：采用校准相机约束的双向稀疏 LK、逐像素局部残差支持和保守遮挡排除；不填充 Box，不把图像证据等同于 Transformer 依赖认证。']
    (ROOT/'diagnosis.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__': main()
