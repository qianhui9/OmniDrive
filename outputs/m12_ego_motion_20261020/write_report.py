"""Render this completed run's report; never overwrite an existing report."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
read = lambda name: json.loads((ROOT / name).read_text())
all_results = read('comparison_final.json')
variants = all_results['variants']
funnel = read('motion_funnel_final.json')
selection = read('run03/frozen_protocol.json')
inventory = read('qualification_inventory.json')
files = read('changed_files_final.json')['files']


def table(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] * len(head)) + '|'] +
                     ['| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in rows])


def percent(v):
    return 'null' if v is None else f'{100*v:.3f}%'


reasons = {
    'ego_motion.py': '新增相机时刻逐点世界投影、原始/初始化 LK、冲突拒绝、条件历史深度和逐射线深度区间。',
    'world_motion.py': '资格及校准位置支持的三帧世界速度/协方差；不足则 Unknown。',
    'geometry.py': '校验 raw K 与预处理映射；合格运动预测进入既有世界几何预测。',
    'config.py': '增加 off/rotation/depth 与资格门控动态预测开关，原接受/风险门限不变。',
    'state_types.py': '增加原始内参、真实运动测量、分开的运动历史及预测记录。',
    'temporal_visibility.py': '预测进入真实 LK；补偿模式使用逐点预测及有界深度区间验证可见性。',
    'identity_temporal.py': '实际身份视图消费来源绑定的历史深度；保留光度/仿射/空间/竞争门；快照绑定 raw K。',
    'foreground_correspondence.py': '动态类显式拒绝未经物体运动补偿的静态三角化证据生产。',
    'association.py': '可靠中心预测及协方差参与门控；3σ 消耗固定 2m 门限预算，不扩大门限。',
    'observation_validator.py': '把真实历史相机和 Anchor 传入既有可见性验证，描述子条件不变。',
    'state_manager.py': '原子提交/消费历史深度、历史摘要、动态排除、场景清空、资格后运动更新及可见性分类。',
    'state_store.py': '事务 fork 复制运动元数据；既有来源摘要覆盖新增字段。',
    'provenance.py': '导出运动原因及深度来源/相机/提交版本，保留神经 Tensor 不可用语义。',
    'input_contract.py': 'Frame 摘要绑定原始内参。',
    'omnidrive_adapter.py': '真实模型输入携带 raw K，保持相机时刻世界位姿与原生推理接线。',
    'dataset.py': '真实 RGB 回放携带原始 K，核查预处理射线一致性。',
    'effective_acceptance.py': '支持预先冻结的显式场景清单，仍逐场景独立运行。',
    'coverage_oracle.py': '仅离线增加 LiDAR 深度、类别和可见性参考字段，不进入在线 M1。',
    'scene_diagnosis.py': '显式选择诊断配置，导出预测类型及可见性原因。',
    'motion_ablation.py': '先冻结元数据极值场景，再执行四组独立回放。',
    'motion_scoring.py': '离线稀疏对应评分、分层、残差和 Micro/Macro；未执行预测字段为 null。',
    'motion_revalidate.py': '无 GT 重建更早真实对应产生的深度，核验帧/相机/版本来源。',
    'motion_cases.py': '从真实迹线画支持/拒绝案例，不绘制失败 LK 占位坐标。',
    'test_ego_motion_compensation.py': '新增 62 项几何真值、动态预测、竞争、遮挡、来源篡改、事务回滚及可视化用例。',
}

parts = ['# OmniDrive M1.2 实现与独立验收',
         '**已完成代码与实测；本轮不满足进入 M2 的前置条件。** M1 为 observe_only，M2/M3 关闭；真实实验正式认证全部为 0，正式错误接受风险及置信上界为 null。没有网络跳算、跨帧神经 Tensor 或 Prefix/KV 复用。',
         table(['验收门', '结论', '范围'], [
             ['Gate_A_Operational', 'PASS', '真实原生调用、事务、来源、生命周期与场景隔离'],
             ['Gate_A_Functional', 'FAIL', '未在多个场景恢复相对 Baseline 的新增历史 Anchor 支持'],
             ['Gate_B', 'BLOCKED', '无合法独立资格，验证日志不足，无正式可评分接受人口'],
             ['Gate_C', 'PASS（接口范围）', '来源/版本/失效/导出和不可用 Tensor 标记；不授权 M2']]),
         '机器结论：[formal_gates.json](formal_gates.json)。']

parts += ['## 已有机制与新增实现',
          '原代码已有 ego→world、每相机时刻的 world_to_camera、世界 Anchor 投影、相对位姿极线检查及双视图测量。本轮增加的是预测进入真实跟踪及逐点可见性验证，没有重写已有世界坐标机制。',
          '采用列向量 SE(3) 与光学 z 深度；K_image=raw_to_image@K_raw。源像素经逆 K、深度和源相机逆位姿到世界，再投影到当前相机。每相机真实时间、空间、K/预处理映射、执行签名均验证。没有深度时只用旋转初始化搜索；平移下深度仍未知，不对所有深度使用全局单应。有限但错误的基线尺度可能无法被极线约束发现，合成反例明确保留这一限制。',
          'B/C 的点深度来自此前成功提交的真实 RGB 双视图测量，经过正深度、视差角、重投影、局部支持及动态/遮挡排除；只供下一帧消费。当前新出现的动态检测也会从下一帧静态深度历史排除。该深度依赖静态射线假设，误差尺度是条件估计，**不是独立深度传感器或度量资格**。静态框逐射线区间也是条件搜索假设，不证明静止、前景归属或精确三维。在线 M1 未读取离线 LiDAR/标注。',
          '保留原始 LK，再执行预测初始化 LK。原 FB≤0.75px、LK error≤12、极线≤1px、光度/梯度和唯一性门保留；两个有效初始化冲突时拒绝。Anchor 仍要求至少 4 点、支持≥0.8、空间覆盖≥0.75、原描述子残差≤0.18和原搜索半径。动态类不再生产未经物体运动补偿的静态三角化物理证据。',
          '动态世界速度要求实际提交、正式身份链和已校准位置不确定性，至少三帧，传播协方差。原始模型速度与可靠速度分开；本次可靠动态预测为 **0/2638 次真实物体提交**。协方差仅额外消耗固定 2m 关联预算，不扩大门限；竞争拒绝不变。合成资格契约夹具不构成资格数据或资格文件。',
          '## 修改文件',
          table(['文件', '原因'], [[f'[{f}](../../{f})', reasons.get(Path(f).name, '独立 M1-only 对照配置：observe_only、无 Tensor 捕获、无资格文件、原门限。')] for f in files]),
          '共 28 个文件；[完整补丁](implementation.patch)、[SHA 清单](changed_files_final.json)。physical_certifier.py、既有风险门限、基础模型/权重/LoRA、Planning 和生成参数未改。既有生产者摘要覆盖 M1 源码和配置，新代码/模式改变协议摘要，旧资格不能静默复用。未修改 M2/M3 源码。',
          '## 真实调用与生命周期',
          '原生 Petr3D.simple_test → frame_from_model（真实 RGB、日志来源、相机时刻位姿）→ WorldStateManager.precheck → 输入/历史来源验证、世界投影 → 原始/初始化 LK 和实际证据 → 逐点可见性/历史观测判定 → 原生完整 Backbone 与检测/地图头 → outputs_from_model → 私有 fork 中关联、身份提案、资格认证、运动/深度历史及 Feature 元数据 → 单一原子发布 → 原始 Planning。没有新增推理执行器。',
          'Precheck 只消费已经提交的图像/深度，当前 Commit 产生的证据只能从下一帧使用。当前认证与历史 Precheck 结果分开导出。场景、Epoch、帧/相机摘要、提交版本和执行签名绑定历史。异常、超时、输入/来源变化回滚或丢弃污染状态；场景/布局/签名变化、时间倒退、重复帧、长间隔保持原重置机制，并清空深度和身份历史。未观测不等于消失；历史可保留到过期，但不能报为本帧已验证。',
          '主四组测量期间代码不变。测量后仅纠正辅助可见性标签：incomplete_depth 归 Unknown，零覆盖归出视野，部分裁剪单列；原投影原因及接受判定没有改变。[精确测量源码](measured_source/)、[更正及 SHA](visibility_audit_correction.json)均保留。交付版重新通过全部 809 测试和一个同场景完整模型开关对照，主要计数一致，见 [delivery_comparison.json](delivery_comparison.json)。']

scenes = variants['c']['operational']['scenes']
by_scene = {v: {s['scene_id']: s for s in variants[v]['operational']['scenes']} for v in variants}
identity = {s['scene_id']: s for s in variants['c']['identity']['scenes']}
features = {s['scene_id']: s for s in selection['metadata_features']}
parts += ['## 选择与执行范围',
          '先按真实位姿及既有检测数量固定不同极值场景：最小/最大平移、最大累计/单步转角、最多/最少物体。没有根据本轮结果或 GT 换场景。[冻结协议](run03/frozen_protocol.json)保留全部场景/日志/帧映射。嵌套计数曾残留可用场景数 38，已以[单独记录](run03/selection_reconciliation.json)更正为实际 6；外层冻结 ID、所有样本和回放协议始终一致。',
          table(['场景', '日志', '12 帧自车平移/累计旋转'], [[s['scene_id'], s['log_token'], f"{features[s['scene_id']]['ego_translation_m']:.3f}m / {features[s['scene_id']]['ego_rotation_deg']:.2f}°"] for s in scenes]),
          '这些场景包括停车场行人、施工车辆、雨天右转、停放卡车/路口、密集交通/移锥和窄路。每场景独立实例、Store、图像、身份/深度历史、计时和目录，每组 72 帧、432 相机输入。两次回放一致，汇总只计 repeat_0，不把重复当新样本。使用连续可用 nuScenes keyframes，实际间隔约 0.400–0.601s，**不是高频视频**；缺帧/重置逐场景记录。',
          'CPU 执行 38 个有检测缓存场景中的 6 个，另 32 个未执行；训练 infos 总 700 场景，未执行 694 个。GPU 执行验证集 150 场景中的 2 个，未执行 148 个。CPU 是真实 RGB+历史完整模型检测回放，不是本次神经推理；GPU 是完整模型与 Planning。GPU 四组全套对照 Not Run，仅执行 C 开/关。专门夜间/光照突变及 GT 遮挡重入事件级召回专项 Not Run；合成测试不能代替真实专项。没有全数据集结论。',
          '## 逐场景结果']
rows = []
for s in scenes:
    sid = s['scene_id']; c = s['counts']; ic = identity[sid]['counts']
    rows.append([sid, c['historical_projection_candidates'], *[by_scene[v][sid]['counts']['observation_supported_anchor_events'] for v in ['baseline', 'a', 'b', 'c']],
                 f"{c['identity_candidate_proposals']}/{c['identity_proposals']}", f"{ic['candidate_correct']}/{ic['candidate_wrong']}/{ic['candidate_unscorable']}"])
parts += [table(['场景', '候选分母', 'Baseline支持', 'A支持', 'B支持', 'C支持', 'C身份提案/总数', '提案正确/错误/不可评分'], rows),
          '以上所有正式认证均为 0。历史支持、有限身份提案、稳定图像点和物理资格是不同人口。五个 CPU 场景支持为零；9a1188…和930329…身份提案全部 Unknown。febc1800…有157个身份提案但历史 Anchor 支持仍为0，不能合并解释。',
          '## Micro、Macro 与独立评分']
rows = []
for v in ['baseline', 'a', 'b', 'c']:
    x = variants[v]['operational']; c = x['micro']['counts']; p = variants[v]['correspondence']['micro']['observation_supported']['counts']; macro = x['macro']['observation_anchor_coverage']
    rows.append([v, f"{c['observation_supported_anchor_events']}/{c['historical_projection_candidates']}", percent(x['micro']['observation_anchor_coverage']), percent(macro['mean']), percent(macro['median']), percent(macro['p95']), f"{p['correct']}/{p['wrong']}/{p['unscorable']}"])
parts += [table(['组', '支持/候选', 'Micro覆盖', 'Macro均值', 'Macro中位数', 'Macro P95', '支持图像点正确/不一致/不可评分'], rows),
          'B/C 恢复 A 丢失的支持但没有超过 Baseline。各组身份提案225/5211；可评分163正确、0错误、62不可评分，其余为Unknown。未知边中存在错误关联，不计为正式错误接受；也不能把163/163外推为总体身份准确率。正式接受人口为空，风险与上界为null。',
          '点评分仅在在线提交后使用稀疏 LiDAR、标注物体刚体位姿及双帧深度可见性参照；目标距离≤1px、对应残差≤2px，竞争表面不可评分。未标注背景静态、刚体框、相机/LiDAR异步均有局限，不是密集实例真值。B/C 支持点增加242，可评分不一致206→208，且约97.7%不可评分，不能宣称可靠性提升。这些点误差不是身份错误接受风险。',
          '每项比率保留分子/分母，零分母为null；各场景深度、类别、自车运动和稀疏可见性分层见 scoring_*_final/summary.json。完整24行：[scene_summary_final.csv](scene_summary_final.csv)；Micro/Macro原始计数和分布：[comparison_final.json](comparison_final.json)。',
          '## 深度证据漏斗与拒绝原因',
          table(['B/C 每次回放人口', '数量'], list(funnel['c']['micro_counts'].items())),
          '4057次条件深度预测中3943次在预测视野内；深度初始化恢复132条FB有效跟踪，100条再过几何，83条保留排除后的局部支持。83个图像点不是83个Anchor或认证。无GT独立核查从更早真实相机/对应重建B、C各8114次深度来源（包含两次重复），不增加独立风险单位。',
          '在原始及选择后均FB有效、预测在视野内的共同人口上，B/C有160757对：相对预测残差中位数0.205px、P95 33.566px；有效原始LK被保留，配对选择后残差相同。选择后FB中位数0.00373px、P95 0.256px。失败原始LK不进入这个配对统计；详见 [motion_funnel_final.json](motion_funnel_final.json)。',
          '1191个几何可用候选相机视图，来源帧/Epoch/版本/签名均通过；历史裁剪缺失147，描述子残差拒绝848，遮挡规则拒绝712。实际可见性最低点数仅97通过，支持比例仅49通过。各检查独立计数，拒绝可能重叠，不能相加。主要限制仍是视角变化后裁剪/描述子、前景和空间支持，以及快速平移时深度/身份不可观测。',
          '[逐帧/Anchor/相机漏斗](diagnosis_c/)、[可见性/失效/拒绝原因](visibility_and_lifecycle_final.json)。可视化：[CPU案例清单](cases_c_final/manifest.json)、[GPU案例清单](cases_gpu_final/manifest.json)。绿/红/橙为稀疏参照正确/不一致/不可评分；青色只画有效原始LK，紫色只画有效预测。不绘制失败占位点，不把图像线段当实例准确率。早期评分/绘图及源码快照保留，最终采用 *_final。',
          '## 完整模型验证']
gpu = read('gpu_c_final/summary.json')
parts += [table(['验证场景', '支持/候选', '身份提案/总数', '正式认证', '原生模型/Planning'], [[s['scene_id'], f"{s['counts']['observation_supported_anchor_events']}/{s['counts']['historical_projection_candidates']}", f"{s['counts']['identity_candidate_proposals']}/{s['counts']['identity_proposals']}", 0, '开/关一致'] for s in gpu['scenes']]),
          'GPU Micro支持45/1935=2.326%，Macro均值/中位数2.168%；第二场景仍为零。身份提案186/1710，离线98正确、0错误、88不可评分。稀疏点984正确、114不一致、50946不可评分。Backbone、检测/地图输出、原生Memory和Planning文本开关一致，源码前后摘要相同。',
          '辅助标签修复后的交付版本又在相同冻结场景运行12帧完整开/关对照，所有计数（45/1038等）与原测量一致，原生输出和Planning一致：[gpu_delivery/final_verification.json](gpu_delivery/final_verification.json)、[ledger_gpu_delivery.json](ledger_gpu_delivery.json)。重复不算新增场景/日志。',
          '## 延迟与CPU开销']
rows = []
for v in ['baseline', 'a', 'b', 'c']:
    lat = variants[v]['operational']['latency_micro']
    rows.append([v, *[f"{lat[k]['p50']:.1f}/{lat[k]['p95']:.1f}" for k in ['precheck_wall_ms', 'commit_wall_ms', 'end_to_end_wall_ms', 'm1_cpu_process_ms']]])
parts += [table(['组', 'Precheck P50/P95 ms', 'Commit P50/P95 ms', '端到端P50/P95 ms', 'M1 CPU P50/P95 ms'], rows),
          'CPU端到端含图像IO/解码、Precheck、Commit和审计，不含本次神经推理及持久化/画图。C相对同帧Baseline配对额外Precheck/Commit中位约76.6/110.7ms，M1进程CPU中位增量301.7ms。并行任务、读盘缓存影响墙钟，不是隔离微基准；C较低端到端P50不是加速证据。[逐场景配对和Micro/Macro](performance_final.json)。',
          'GPU完整模型C端到端P50/P95=4762.5/7633.9ms，Precheck=483.4/990.7ms、Commit=728.8/2996.0ms，包含完整模型/M1/Planning、排除batch IO，不能与CPU检测缓存回放直接比较。没有GPU跳算/神经复用收益声明。',
          '## 测试与资格',
          '**交付版809通过，0失败，0跳过**：M1 496、M2 127、M3 131、跨模块55。旧747项与本轮新增62项分别计数，重复运行不累加。[实际日志](regression_delivery.log)、[JUnit](regression_delivery.xml)、[每项用例](test_results_delivery.json)。',
          '新增测试含可计算几何真值：纯平移/旋转/组合、不同深度视差、静态点；真实纹理光流、低纹理；时间/位姿/K/预处理错误；来源/版本/签名篡改；原子失败回滚；遮挡保持、重入、过期；0/1/15m/s和转向不确定性；自车/物体同时运动、交叉目标竞争；可视化不画失败占位。合成数据及资格契约夹具未进入在线决策或风险人口。真实模型开/关对照另行验证原生Planning/计算不变。',
          '四组各12条重复场景迹线（共48条）及GPU两条独立来源/版本核查通过；B/C还重建更早实际深度来源。[ledger_b.json](ledger_b.json)、[ledger_c.json](ledger_c.json)。这些工程核查不是物理资格。',
          f"既定验证集{inventory['frames']}帧、{inventory['scenes']}场景、{inventory['independent_logs']}独立日志；按日志分校准后仅{inventory['validation_log_units']}个可用验证日志，假设零错误的95%上界仍是{inventory['best_case_zero_error_upper_95']:.6f}>0.05。至少需要59个零错误独立验证单位。实际正式可评分接受日志是0，风险和上界为null。场景、帧、重复或已暴露训练数据不能冒充独立验证日志。未生成生产资格文件。", 
          '历史M1.1的CPU六场景与本轮不同，不能用旧汇总0对新39宣称改进；只比较本轮固定Baseline/A/B/C。相同两个GPU场景主要支持/身份提案计数与M1.1一致，也不是新增收益。旧实验目录未覆盖。首轮YAML off解析错误在处理任何帧前被拒绝；run02及首次GPU权重校验中止记录保留，不计验收。早期沙箱回归的1项skip不计入最终809全通过结论。',
          '## 复现与剩余最小修复',
          '在项目根目录使用新的目录：\n\n```bash\ntaskset -c 0-3,6-31 bash outputs/m12_ego_motion_20261020/reproduce.sh outputs/m12_reproduction_NEW\n```',
          '脚本包含OmniDrive解释器、冻结场景、四组两次回放、离线评分及独立日志划分、来源重建、回归、完整模型和案例导出。GPU忙只等待，不停止其他进程。当前代码会输出修正的辅助标签及新协议摘要；精确原测量源码/配置已归档，可在独立副本中复现原字节；计时不保证逐次相同。',
          '下一步最小M1修复：针对已定位的裁剪/描述子及前景支持失配，在绑定真实对应上验证局部深度对齐/源裁剪重建，保持现有残差、空间、遮挡和竞争门，先以留出测量证明错误不增加。快速物体仍需合法身份链和独立位置不确定性校准；资格仍需足够未暴露日志及可评分接受人口。目前这些条件未成立，保持Unknown，不进入M2。']

out = ROOT / 'experiment_report.md'
if out.exists():
    raise ValueError('Do not overwrite the report')
out.write_text('\n\n'.join(parts) + '\n')
print(out)
