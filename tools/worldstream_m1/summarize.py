"""Produce the delivery report directly from measured artifacts, without invented values."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .io_utils import write_json
from worldstream.m1 import M1Result


def percent(value):
    return 'N/A' if value is None else '{:.2f}%'.format(value*100)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='outputs/module1')
    args = p.parse_args()
    root = Path(args.root)
    def read(path): return json.loads((root/path).read_text())
    tests = ET.parse(root/'unit_tests.xml').find('testsuite')
    pipeline = read('real_pipeline_smoke.json')
    gpu_path = root/'model_smoke/results.json'
    gpu = json.loads(gpu_path.read_text()) if gpu_path.is_file() else None
    gpu_passed = bool(gpu and gpu.get('observe_only_equal'))
    gpu_count = len(gpu['baseline']['outputs']) if gpu_passed else 0
    gpu_summary = ('- RTX 3090 完整模型 {} 帧 Baseline/Observe-Only：通过；Backbone、Detection/Map 输出及 Temporal Memory 数值摘要完全一致。'.format(gpu_count)
                   if gpu_passed else '- GPU 完整 Baseline/Observe-Only 尚未通过；状态见 model_smoke/status.json。'
                   '宿主驱动正常，受限执行环境可能隐藏 /dev/nvidia*，不能据此断言驱动不可用。')
    lines = ['# M1 实测交付记录', '',
        '关联/有效性实验使用真实 nuScenes 多视图关键帧及历史完整检测回放，Map 回放使用 OpenLane 标注辅助 commit。'
        + ('另外已执行新的 GPU 完整模型对照。' if gpu_passed else 'GPU 完整模型对照状态见下文。')
        + 'M1 没有跳过任何计算。', '',
        '## 验证', '',
        '- 单元测试：{} 项，失败 {}，错误 {}，跳过 {}。'.format(tests.attrib['tests'], tests.attrib['failures'], tests.attrib['errors'], tests.attrib['skipped']),
        '- 真实 CPU 图像 Pipeline：{}，Tensor shape={}，输入未被适配器修改。'.format(pipeline['status'], pipeline['image_tensor_shape']),
        '- 最终预处理投影与项目 lidar2img 最大绝对差 {:.3g} px（双精度、同一标定路径数值交叉验证，不代表实际标定误差）。'.format(max(pipeline['processed_lidar2img_projection_error_px'].values())),
        gpu_summary, '',
        '## 首场景物理关联', '',
        '| 方法 | Precision（可评分对） | Recall | ID Switch | 可评分/全部提议 | 重投影误差 px | 回放计时 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in read('replay_strict/association_metrics.json')['methods'].items():
        lines.append('| {} | {} | {} | {} | {}/{} | {:.3f} | {:.3f} |'.format(name,
            percent(r['precision']), percent(r['recall']), r['id_switches'], r['scorable'], r['predicted'],
            r['mean_reprojection_error_px'], r['mean_runtime_ms']))
    lines += ['', '未知身份未充当误匹配。ID Switch 使用本回放定义，不是官方 tracking benchmark。关联回放时间受到 tracemalloc 影响；性能结论用下方独立 profile。', '',
        '## 首场景状态有效性', '',
        '| 配置/方法 | 候选率（全部历史试验） | 物理 Oracle 错误接受率 | 有效状态恢复率 | 被接受且有标签 |',
        '|---|---:|---:|---:|---:|']
    for run, modes in [('replay_strict', ['geometry_only','appearance_only','geometry_observation']),
                       ('replay_appearance_ablation', ['geometry_observation'])]:
        data = read(run+'/validity_metrics.json')
        for mode in modes:
            r = data['methods'][mode]
            lines.append('| {}/{} | {} | {} | {} | {} |'.format(run, mode,
                percent(r['valid_candidate_rate']), percent(r['false_reuse_rate_physical_oracle']),
                percent(r['valid_state_recovery_rate']), r['accepted_labeled']))
    lines += ['', '首场景共 {} 个历史状态试验，{} 个有可用 Oracle 身份。严格认证没有独立可见性证据，因此没有接受状态；其 False-Reuse 是 N/A，不能写成 0%。消融仅取消可见性要求，其错误率说明不能将它当作安全缓存规则。Oracle 只验证物理中心/身份/可见性，不验证深层特征。'.format(
        data['historical_anchor_trials'], data['methods']['geometry_observation']['labeled_count']), '',
        '## 真实场景覆盖', '', '| 运行 | 帧数 | 实际帧间隔范围 s | 历史状态试验 | 新入相机视野标注区域覆盖 | 未知背景更新率 |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ['replay_strict','replay_second_scene','replay_interval_1s']:
        manifest = read(name+'/manifest.json'); metrics = read(name+'/validity_metrics.json'); novelty = read(name+'/novelty_metrics.json')
        intervals = manifest['intervals_s']
        lines.append('| {} | {} | {:.4f}–{:.4f} | {} | {}/{} | {} |'.format(name, len(manifest['samples']),
            min(intervals), max(intervals), metrics['historical_anchor_trials'],
            novelty['covered_new_gt_regions'], novelty['new_gt_regions'], percent(novelty['uncovered_background_update_ratio'])))
    lines += ['', '全图未解释时的覆盖召回为平凡结果，意味着请求完整更新，不意味着语义新物体检测成功。开放世界 semantic FPR 缺少可靠负例，保持 null。', '',
        '## CPU 性能与内存', '',
        '| Anchor 数 | 描述 | Projection ms | Observation ms | Unexplained ms | Association ms | State Update ms | Total ms | Traced peak MiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in read('profile.json')['results']:
        t = r['mean_cpu_wall_ms']
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(
            r['anchor_limit'], r['descriptor'], t['projection'], t['observation_validation'], t['unexplained_regions'],
            t['association'], t['state_update'], t['total'], r['state_and_temporary_traced_peak_bytes']/1024**2))
    lines += ['', '真实两帧微基准，warmup=1、repeats=5；分配跟踪另测。以上不含完整模型、离线 Oracle、IO/可视化或在线 GPU→CPU 输入拷贝。默认未保留任何大型 Tensor，retained_tensor_bytes=0；Anchor 的小描述仍占用 CPU 内存。该 CPU 微基准不测 GPU 时间/显存；完整模型指标单列。进程 RSS 含输入表，不用作 M1 净内存。', '',
        '## 轻量描述对照与校准', '', '| 描述 | 固定 crop 外观错误接受率 | 人工亮度扰动平均残差 |', '|---|---:|---:|']
    for name in ['replay_strict','replay_gradient']:
        manifest = read(name+'/manifest.json'); metrics = read(name+'/validity_metrics.json')
        rows = [json.loads(r) for r in (root/name/'oracle_records.jsonl').read_text().splitlines()]
        residuals = [r['illumination_residual'] for r in rows if r.get('illumination_residual') is not None]
        lines.append('| {} | {} | {:.4f} |'.format(manifest['config']['descriptor'],
            percent(metrics['methods']['appearance_only']['false_reuse_rate_physical_oracle']), float(np.mean(residuals))))
    calibration = read('calibration.json')
    lines += ['', '亮度对照是在真实当前图像上施加 0.7×I+25，再比较固定区域描述，不代表跨天气泛化。当前结果不足以选择所谓最优阈值；默认灰度实现保持 provisional。', '',
        'train-only 校准状态：{}，独立 Anchor 可用试验 {}，threshold={}。未将评估集用于阈值调节，未写入不满足风险支持的阈值。'.format(
            calibration['status'], calibration['labeled_eligible'], calibration['threshold']), '',
        '## Map 与可视化', '']
    rows = read('map_annotation_demo/results.json')['frames']
    lines.append('真实 OpenLane 标注辅助 Map 回放 {} 帧：每帧地图线数 {}，后续几何匹配数 {}。未声称完成新的 Map Head 推理或官方 Map ID 评价。'.format(
        len(rows), [r['current_lines'] for r in rows], [r['matched_historical_lines'] for r in rows[1:]]))
    lines += ['', '逐帧六视图见各运行目录 visualizations/*.jpg，精确 mask 见 masks/*.npz；contact_sheet.jpg 为连续关键帧总览。figures/*.png 和 *.pdf 为可导出图表。颜色区分历史 crop、预测投影、认证结果、当前完整输出和失效/未解释区域。', '',
        'M2 接口实例见 m2_example.json，字段来自真实回放的最后一次 precheck；valid_feature_candidates 只代表物理/观测候选，Transformer dependency 未认证。完整接口、已验证命令、文件职责、坐标与限制见 ../../docs/worldstream_m1.md。']
    if gpu_passed:
        lines += ['', '## 新执行的 RTX 3090 完整模型对照', '',
                  '| 模式 | 帧数 | CUDA Event 总时长 ms（含 host 间隙） | 峰值 allocated GiB |',
                  '|---|---:|---:|---:|']
        for mode in ('baseline', 'observe_only'):
            r = gpu[mode]
            lines.append('| {} | {} | {:.3f} | {:.3f} |'.format(mode, len(r['outputs']),
                r['gpu_elapsed_ms_including_host_gaps'], r['peak_allocated_bytes']/1024**3))
        lines += ['', '单次、未充分预热的完整执行检查，包含输入处理、数值摘要拷贝与 LLM 生成，不用于推导加速比。'
                  '验证结论是开启 M1 未改变这 {} 帧的视觉/感知/历史 Memory 数值；不要求采样文本逐字一致。'.format(gpu_count)]
        states_path = root/'model_smoke/m1_state.jsonl'
        if states_path.is_file():
            states = [json.loads(row) for row in states_path.read_text().splitlines()]
            costs = [row['commit'].get('m1_total_including_adapters_ms') for row in states]
            costs = [value for value in costs if value is not None]
            lines += ['', 'M1 实际在线 CPU wall overhead（包含输入/输出适配与必要同步拷贝）：{} ms；均值 {:.3f} ms。'.format(
                [round(value, 3) for value in costs], float(np.mean(costs))), '',
                '实际捕获的形状元数据（默认不保留大型 Tensor）：', '', '| 特征 | Shape |', '|---|---|']
            observed = {}
            for row in states:
                for key, value in row['precheck']['feature_versions'].items():
                    if key in ('backbone', 'object_carrier', 'map_carrier'): observed[key] = value['shape']
            lines.extend('| {} | {} |'.format(key, value) for key, value in observed.items())
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    last = json.loads((root/'replay_strict/state.jsonl').read_text().splitlines()[-1])['precheck']
    write_json(root/'m2_example.json', M1Result(**last).to_dict())
    paths = list(Path('worldstream').rglob('*.py')) + list(Path('tools/worldstream_m1').glob('*.py')) + list(Path('tests/worldstream_m1').glob('*.py'))
    paths += [Path('configs/worldstream_m1.yaml'), Path('projects/configs/OmniDrive/worldstream_m1_observe.py'),
              Path('projects/mmdet3d_plugin/models/detectors/petr3d.py')]
    write_json(root/'source_sha256.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
    print(root/'experiment_report.md')


if __name__ == '__main__': main()
