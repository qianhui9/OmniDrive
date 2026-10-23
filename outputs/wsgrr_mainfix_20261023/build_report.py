"""Populate the final human report from completed, preserved experiment records."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
d=read(ROOT/'summary.json');o=d['official'];full=d['full_execution'];b=read(ROOT/'before_full_execution.json')
assert read(ROOT/'final_evidence.json')['passed']
assert read(ROOT/'finish_status.json')['phase']=='COMPLETED'
report=ROOT/'experiment_report.md';text=report.read_text()
start=text.index('当前状态：') if '当前状态：' in text else text.index('最终状态：')
end=text.index('\n\n',start)
text=text[:start]+('最终状态：**冻结的唯一 WSGRR 主方案已完成 6019 帧、150 场景官方 Detection/Map 评估，'
    '以及两次同输入完整请求复测。Detection 精度损失显著缩小，视觉与完整请求均未获得净加速；未通过生产认证。** '
    '完整机器结果见 `summary.json`，来源与公平性核验见 `final_evidence.json`。')+text[end:]
text=text.replace('本轮新真实预测的逐对象/逐线差异，须待 `main_full/predictions.pkl` 完整生成后追加，不能用 Native 回放代替候选预测。',
    '本轮已从完整真实预测生成 `anomaly_prediction_diagnostics.json`，没有使用 Native 回放冒充候选任务预测。')
needle='## 性能归因的边界'
anomaly='''异常汽车的返回索引 4 在旧正式 D3 中相对 Native 的中心偏移为 **0.625975 m**（先前短测约 0.6253 m），新主方案降至 **0.000857 m**。原生、旧 D3、新主方案至最近真实汽车标注的 XY 距离分别为 **2.960206、3.563052、2.960708 m**；因此只能说额外偏移被显著抑制，不能说这个对象已被正确检测。最近 GT 不是对象身份认证，也不替代正式分数排序匹配。

旧 Map 51.792896 m 来自 Native 线 95 与候选线 87 的强制配对。旧候选返回索引 95 的分数为 0.277802，低于原有 0.3 诊断截断，而其与 Native 线 95 的对称几何距离仅 0.138963 m。新主方案该分数恢复至 0.367388、对称距离为 0.003458 m；同帧旧口径最大匹配距离为 0.073762 m，新门控诊断匹配 23 条，原生未匹配 0 条、候选未匹配 1 条。旧 95→87 配对在新预测中仍空间无关，其对称距离 29.939454 m 也保留用于复核。**历史异常没有删除；诊断分数、2m/3m 门限及正式评估口径均未调整。** 同返回索引本身不证明身份，诊断一致性不等于 GT 精度。

'''
if '异常汽车的返回索引' not in text:text=text.replace(needle,anomaly+needle)

lines=['## 正式 Detection、Map 结果','',
    '完整验证集 6019 帧、150 场景，样本顺序、预处理、任务头和 Detection 评估条件与 Native 一致。'
    'Native 引用通过哈希核验的历史完整结果；本轮只有一个最终完成的 WSGRR 主方案全量评分。',
    '', '| 指标 | Native | 修改前 D3 | 修改后主方案 | 相对 Native 绝对差 | 相对变化 |',
    '|---|---:|---:|---:|---:|---:|']
for key in ['mAP','NDS','mATE','mASE','mAOE','mAVE','mAAE','DET_l','lane_Fscore']:
    delta=o['after_vs_native'][key]
    lines.append(f"| {key} | {o['native'][key]:.6f} | {o['before'][key]:.6f} | {o['after'][key]:.6f} | {delta['absolute']:+.6f} | {delta['relative_percent']:+.4f}% |")
lines+=['','mATE、mASE、mAOE、mAVE、mAAE 越低越好。Detection mAP 的相对损失由约 10.48% 缩小到 **0.4351%**，NDS 由约 6.53% 缩小到 **0.2652%**。'
    'Map 没有全面提升：DET_l 比旧 D3 下降 0.003414，车道 F-score 比旧 D3 增加 0.000267；两者目前均接近 Native。',
    '', '模型实际支持车道中心线 DET_l 与 3D Lane F-score。没有交通元素或学习到的拓扑预测头，零邻接占位产生的 TOP_ll 等值不能解释为拓扑能力。',
    '', '| 类别 AP | Native | 修改前 | 修改后 | 对 Native 相对变化 |', '|---|---:|---:|---:|---:|']
for key,v in o['per_class'].items():
    lines.append(f"| {key} | {v['native']:.6f} | {v['before']:.6f} | {v['after']:.6f} | {v['after_vs_native']['relative_percent']:+.4f}% |")
lines+=['', '150 个场景中，134 个比旧 D3 的 mAP 更高；90 个仍低于 Native、60 个高于 Native。'
    '场景 mAP 差值中位数为 -0.000407、最差为 -0.023656。逐场景结果见 `paired_scenes.json`。',
    '', '| 残余下降场景 | Native mAP | 旧 D3 | 新主方案 | 场景描述 |', '|---|---:|---:|---:|---|',
    '| scene-0797 | 0.182737 | 0.128489 | 0.159080 | 汽车、公交经过、超车、等待 |',
    '| scene-0912 | 0.210489 | 0.135991 | 0.190786 | 雨天、停车场、卡车、工程车辆 |',
    '| scene-1067 | 0.089918 | 0.050793 | 0.071452 | 夜间雨天、暗处行人、进入交叉口 |',
    '', '正式 SDK 2m 分数排序匹配的原始统计（全部保留分数，无 0.3 诊断截断）：Native/新主方案预测数为 1,283,894/1,285,217，TP 为 112,054/111,966，FN 为 9,807/9,895，GT 总数均为 121,861。'
    '这是原始匹配工作点统计，不是 mAP 的替代指标。逐类、逐对象和漏匹配记录位于 `main_full/official_eval/detection_matches_2m/`。',
    '', '## 全量窗口与缓存有效性', '', '| 6019 帧统计 | 修改前 | 修改后 |', '|---|---:|---:|']
for label,key,percentage in [('执行更新率','update_rate',True),('复用率','reuse_rate',True),('执行窗口数，两个缓存 Block','updated_windows',False),
    ('必刷唯一窗口数','mandatory_windows',False),('普通唯一窗口数','ordinary_windows',False),('已识别必刷集合执行遗漏','recognized_mandatory_misses',False),('缓存发布失败','cache_publication_errors',False)]:
    a,c=b[key],full[key]
    lines.append(f'| {label} | {a*100:.4f}% | {c*100:.4f}% |' if percentage else f'| {label} | {a:,} | {c:,} |')
lines+=['', '两个 Block 的执行窗口分母均为 650,052；必刷与普通数量按每帧唯一相机窗口计数，不能与双 Block 工作量直接相加。'
    '新主方案普通预算超限 0 次、未来来源时间戳 0 次；637 帧按 M1 安全约束完整刷新。'
    '这验证已声明约束被执行，不表示真实事件漏检率为零，也不构成 M1 世界身份认证。',
    '', '## 完整请求与性能结论', '',
    '以下是相同 40 帧、40 个唯一问题，两次重复的 80 个观察值；顺序为 Native_r0 → main_r0 → main_r1 → Native_r1。'
    '每次预热 1 帧后重置状态，正式计时包含冷启动首帧和后续复用，GPU 在边界同步。'
    '包括预处理、传输、M1、M2、Detection、Map、原生语言生成与输出处理，审计序列化和计时外检查不计入。'
    '两次重复不能当作 80 个独立场景。', '',
    '| 完整请求 ms | 历史 Native | 历史 D3 | 本轮 Native | 本轮主方案 |', '|---|---:|---:|---:|---:|',
    '| P50 | 1708.80 | 2106.76 | 1714.24 | 2060.36 |', '| P95 | 9108.50 | 8106.82 | 9081.68 | 8941.24 |',
    '', '| 本轮重复 | Native P50/P95 ms | 主方案 P50/P95 ms |', '|---|---:|---:|',
    '| 0 | 1832.64 / 9089.09 | 2030.36 / 8915.78 |', '| 1 | 1676.80 / 9062.91 | 2086.89 / 8942.80 |',
    '', '**视觉 P50 比同轮 Native 慢 1.17%；完整请求 P50 慢 20.19%，平均耗时慢约 10.62%。未获得稳定净加速。** '
    '完整请求 P95 略低不能当作视觉代码加速：Native/主方案每次实际生成 3022/2926 个 token，真实输出长度不同。'
    '输入问题、参考答案、输入 token 数、生成参数与 Native 生成路径相同，核验见 `final_evidence.json`、`vlm_groups.json`。',
    '', '当前完整请求中 M1 Precheck 主机 P50 为 195.61 ms，M1 Commit 主机 P50 为 76.56 ms，含适配器的 M1 总量 P50 为 292.13 ms，M1–M2 状态读取 P50 为 25.16 ms。'
    '这些范围嵌套，不能相加。历史相同范围的完整 M1 总量未记录，未用猜测补值；可比记录见 `request_phase_breakdown.json`。'
    'CUDA event 的 stream span 包含 CPU 提交间隙和等待，不等于活跃 GPU 内核时间。',
    '', '全量感知请求描述性 P50 为 1002.95 ms，其中 M2 诊断墙钟 P50 为 487.29 ms。'
    '该全量计时没有本轮重算的配对 Native，不能拿它与历史 Native 计时直接宣称速度变化；净收益判断使用本轮同步配对测量。',
    '', '## VLM 短测与未完成项', '',
    '40 个唯一问题的 Exact Match 均为 0；Token F1：Native 0.359270，主方案 0.369617，两次重复结果相同。'
    '9 个含数值参考的问题均未满足预测与参考的数字数量一致条件，数值 MAE 的有效分母为 0，应报告不可用，不能填 0。'
    '空间关系 5 题 Token F1 均为 0.343176；global_safety 15 题为 Native 0.368056、主方案 0.365157。'
    '分组来自既有问题分类器，Exact Match 的失败率不是独立标注的驾驶语义错误率。',
    '', '完整 VLM 验证集、独立语义正确性/关键驾驶场景错误率尚未验证；不能用这 40 题外推。'
    'Map 交通元素和学习拓扑指标因模型缺少对应预测头不适用。',
    '', '## 验收判断与保留限制', '',
    '关键图像风险导出遗漏与旧范围撤销已修复，预算、年龄、来源与依赖约束通过回归和连续审计；正式 Detection 损失显著改善。'
    '但没有预先批准的生产精度容忍阈值，且部分场景仍有明显下降、Native 自身也有错误，不能自动认定质量可接受。'
    '部分窗口执行器的既有浮点差异仍在，M1 总开销与全局后缀成本仍大，净加速目标未达成。生产批准表和正式配置均未修改。',
    '', '开发过程中一次未完成推理在 1220 帧中止以补齐空框/越界/非有限范围撤销；记录保存在 `main_full_pre_boundaryfix/`，没有正式评分，未用于最终指标。'
    '本报告全量结果均来自最终冻结代码的完整 `main_full/`；其他开发失败/中断见 `attempts.json`。', '']
residual=read(ROOT/'residual_detection_cases.json')
part=['### 残余对象的实际预测复核','',
    '按“Native 是 TP、新主方案成为 FN”集合中 Native 分数最高的样本，各取一个重点类别进行离线复核。'
    '所有该类 GT 状态变化保存在 `residual_detection_gt_changes.json`，没有为了降低误差删样本。'
    '具体坐标、分数、逐窗口年龄/漂移/不确定性、风险记录和执行窗口见 `residual_detection_cases.json`。',
    '', '| 类别 / 帧 ID | Native 对 GT 距离 m | 新方案最近同类候选对 GT 距离 m | 当前缓存前缀全部刷新 |',
    '|---|---:|---:|---|']
for c,v in residual['cases'].items():
    part.append(f"| {c} / {v['frame_id']} | {v['native_matched_prediction']['xy_distance_m']:.6f} | {v['candidate_nearest_GT_xy_distance_m']:.6f} | {'是' if v['all_current_cached_prefix_windows_fresh'] else '否'} |")
part+=['', '汽车、自行车、摩托车的代表性候选仍在原有严格 2m 门限之外；没有放宽门限将其改报为匹配。'
    '行人最近候选距该 GT 为 0.712283m，但已被分数排序匹配分配给另一个 GT；该帧全部缓存前缀都刷新，因此不能单凭当前帧区域遗漏解释。'
    '下游历史状态和匹配竞争仍可能贡献残余差异。其余案例存在局部复用，但窗口元数据本身不构成对象级因果证明；本轮没有新增反事实推理实验。', '']
where=lines.index('## 全量窗口与缓存有效性');lines[where:where]=part
start=text.index('## 正式 Detection、Map 结果') if '## 正式 Detection、Map 结果' in text else text.index('## 正式精度与完整请求')
end=text.index('## 仪器化变更记录',start)
text=text[:start]+'\n'.join(lines)+'\n'+text[end:]
profile={}
for name,path in [('before',ROOT.parent/'wsgrrr_20261022_followup/visual40_final'),('after',ROOT/'visual40_main')]:
    profile[name]={r['name']:r for r in read(path/'profile_output/operators.json')['rows']}
stages=[('Patch Embedding + Block0 输入','m2/block0_input'),('M1-M2 准入读取','m2/computation_admission'),
    ('评分与选择','m2/select_regions'),('依赖计划','m2/dependency_plan'),
    ('Block0 QKV','prefix/0/qkv'),('Block0 Attention','prefix/0/attention'),
    ('Block0 输出投影','prefix/0/output_projection'),('Block0 MLP','prefix/0/mlp'),
    ('Block1 QKV','prefix/1/qkv'),('Block1 Attention','prefix/1/attention'),
    ('Block1 输出投影','prefix/1/output_projection'),('Block1 MLP','prefix/1/mlp'),
    ('缓存原子提交包装器','cache/commit'),('CUDA 流同步等待','cudaStreamSynchronize'),
    ('全路径张量 copy，非缓存专属','aten::copy_')]
table=['<!-- PROFILE_TABLE -->','',
    '| 单帧剖析范围 ms | 修改前 CPU总计 | 修改前归属CUDA | 修改后 CPU总计 | 修改后归属CUDA |',
    '|---|---:|---:|---:|---:|']
for label,key in stages:
    values=[]
    for name in ['before','after']:
        r=profile[name].get(key)
        values+=['—','—'] if r is None else [f"{r['cpu_total_us']/1000:.3f}",f"{r['cuda_total_us']/1000:.3f}"]
    table.append('| '+label+' | '+' | '.join(values)+' |')
table+=['', '以上各行存在父子归属重叠，禁止直接相加；copy 包含后续网络布局转换。同步等待的 CPU 时间不应归为 CUDA 计算。'
    '阶段剖析的选窗工作量不同，只作归因，净速度以上述配对同步墙钟为准。', '', '<!-- END_PROFILE_TABLE -->','']
if '<!-- PROFILE_TABLE -->' in text:
    a=text.index('<!-- PROFILE_TABLE -->');b=text.index('<!-- END_PROFILE_TABLE -->',a)+len('<!-- END_PROFILE_TABLE -->')
    text=text[:a]+text[b:]
text=text.replace('## 正式 Detection、Map 结果','\n'.join(table)+'\n## 正式 Detection、Map 结果',1)
text=text.replace('本目录 `summarize.py`、`finish_validation.py`、`verify_native_reuse.py`、`plot_anomaly_windows.py`',
    '本目录 `summarize.py`、`finish_validation.py`、`run_main_validation.py`、`verify_native_reuse.py`、`plot_anomaly_windows.py`、`diagnose_final_anomaly.py`、`final_evidence.py`、`build_report.py`、`residual_detection_audit.py`')
text=text.replace('离线主方案限制、计时外缓存审计、单帧 cProfile 和分阶段 torch.profiler 调用',
    '离线主方案限制（532行）、计时外缓存审计（140行）、单帧 cProfile（238行）和分阶段 torch.profiler 调用')
if '`residual_detection_audit.py`' not in text:
    text=text.replace('`build_report.py`','`build_report.py`、`residual_detection_audit.py`')
report.write_text(text)
print('Final report:',report)
