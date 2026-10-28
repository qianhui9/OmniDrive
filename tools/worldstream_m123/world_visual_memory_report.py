"""Archive WRVMP stage decisions without upgrading microbenchmarks to E2E."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

from tools.worldstream_m123.benchmark.common import write, sha256
from tools.worldstream_m123.world_visual_memory_analysis import distribution


def assert_training_ready(root):
    """Refuse before any optimizer/model/artifact is created."""
    for file in ('bridge_analysis.json','memory_semantic_probe.json','prefix_cost_feasibility.json'):
        if json.loads((Path(root)/file).read_text())['status'] != 'PASS':
            raise ValueError('WRVMP training refused: '+file+' did not pass')


def generate(root):
    bridge=json.loads((root/'bridge_analysis.json').read_text())
    semantic=json.loads((root/'memory_semantic_probe.json').read_text())
    content=json.loads((root/'feature_content_diagnostics.json').read_text())
    content['semantic_gate']=semantic['status']
    content['semantic_gate_source']='memory_semantic_probe.json; raw feature pretraining probe only'
    write(root/'feature_content_diagnostics.json',content)
    lifecycle=json.loads((root/'natural_audit/memory_lifecycle.json').read_text())['events']
    natural=[json.loads(l) for l in (root/'natural_audit/bridge_capture/frames.jsonl').open()]
    transactions=json.loads((root/'memory_transaction_cost.json').read_text())
    update=json.loads((root/'natural_memory_update_cost.json').read_text())['records']
    controls=[json.loads(l) for l in (root/'controlled_cost/records.jsonl').open()]
    text_controls=[json.loads(l) for l in (root/'controlled_text_cost/records.jsonl').open()]
    rng=np.random.default_rng(20261028)
    def paired(records,left,right,k,metric):
        index={(r['question_index'],r['repeat'],r['kind']):r for r in records if r['K']==k}
        return [r[metric]-index[(r['question_index'],r['repeat'],right)][metric]
            for r in records if r['K']==k and r['kind']==left]
    def timing_summary(values):
        array=np.asarray(values).reshape(3,9).mean(0)
        means=array[rng.integers(0,len(array),(5000,len(array)))].mean(1)
        return dict(distribution(values), timing_bootstrap_CI95_ms=np.quantile(means,[.025,.975]).tolist(),
            CI_scope='Resample nine hardware repetitions, grouping the three questions; one real frame pair. Not scene/population/E2E confidence.')
    cost_table={}
    for k in (64,128,256):
        records=[r for r in controls if r['K']==k]
        source=[r for r in transactions['records'] if r['K']==k]
        cost_table[str(k)]=dict(current_native_tokens=513,total_visual_tokens=513+k,
            hypothetical_fixed513_current_tokens=513-k,
            hypothetical_fixed513_removed_native_tokens=k,
            fixed513_compressed_layout='NOT_IMPLEMENTED_OR_QUALITY_VALIDATED',
            by_kind={kind:{metric:distribution([r[metric] for r in records if r['kind']==kind])
                for metric in ('model_prefill_cuda_ms','executor_wall_ms','matcher_wall_ms','planner_wall_ms')}
                for kind in ('native','memory_zero','memory_exact')},
            K_only_prefill_cuda_saved=timing_summary(paired(text_controls,'memory_text_only','memory_exact',k,'model_prefill_cuda_ms')),
            zeroKV_minus_exact_wall=timing_summary(paired(controls,'memory_zero','memory_exact',k,'executor_wall_ms')),
            native_minus_exact_wall=timing_summary(paired(controls,'native','memory_exact',k,'executor_wall_ms')),
            memory_transaction={kind:{metric:distribution([r[kind][metric] for r in source])
                for metric in ('wall_ms','cuda_stream_ms')} for kind in ('create','keep','projection')},
            natural_valid_frames=sum(e['memory_valid'] for e in lifecycle[str(k)]),
            natural_extensions=sum(e['event']=='keep' for e in lifecycle[str(k)]))
    uby={r['frame_id']:np.mean([x['wall_ms'] for x in update if x['frame_id']==r['frame_id']]) for r in natural}
    overhead=[r['bridge_total_wall_ms']+uby[r['frame_id']] for r in natural]
    scenes=sorted({r['scene_id'] for r in natural})
    scene_costs=np.array([np.mean([cost for r,cost in zip(natural,overhead) if r['scene_id']==scene]) for scene in scenes])
    boot=scene_costs[rng.integers(0,len(scenes),(5000,len(scenes)))].mean(1)
    no_memory=all(not e['memory_valid'] for es in lifecycle.values() for e in es)
    # This experiment's conditional stop is deliberate. Do not invent a PASS
    # result or start a training campaign if future data changes this assumption.
    if not no_memory:
        raise ValueError('Natural evidence changed: implement/review full event-weighted costs before deciding training')
    cost=dict(status='NO_GO',reason='No natural K64/128/256 memory opportunities; complete-current513 append layout is also slower than Native on every measured hit control.',
        selected_layout='Historical regional prefix + complete current513; diagnostic only, no formal adapter',
        main_candidate_K=None,K_diagnostics=cost_table,
        natural_frames=24,natural_scenes=3,natural_qualified_object_frames=6,
        natural_qualified_region_cells=[r['cells'] for f in natural for r in f['regions'] if r['qualified']],
        natural_extensions_by_K={k:sum(e['event']=='keep' for e in es) for k,es in lifecycle.items()},
        world_only_expected_saved_prefill_ms=0.,
        incremental_world_method_net_upper_ms_per_request={
            'one_question':-float(np.mean(overhead)), 'three_questions':-float(np.mean(overhead))/3},
        incremental_cost_scene_bootstrap_CI95_ms=(-np.quantile(boot,[.975,.025])).tolist(),
        CI_scope='Observed three natural development logs only; not population or formal E2E CI.',
        natural_bridge_wall_ms=distribution([r['bridge_total_wall_ms'] for r in natural]),
        natural_memory_update_wall_ms=distribution([uby[r['frame_id']] for r in natural]),
        added_projection_matcher_planner_KV_cache_reassembly_sync_ms=0.,
        omitted_cost_scope='Optimistic incremental upper bound on this zero-memory playback: no memory projection, no memory KV/cache ownership, no assembly are executed. All unmeasured runtime/binding/allocator costs set to zero, never counted as measured. Generic existing text/same-frame caching held equal.',
        old_K64_net_upper_ms=json.loads(Path('outputs/m3_sf_wrcps_feasibility_run01/delivery.json').read_text())['optimistic_one_question_net_ms'],
        beats_old_K64=False,LLM_quality='NOT_RUN',full_E2E_speedup=None,
        alternative_15_to64_learned_resampler='NOT_IMPLEMENTED: could convey 15 real regions to64 learned slots, but cannot be counted as64 independently observed cells; requires new layout/content/cost/quality evidence. This result does not prove that alternative impossible.')
    write(root/'prefix_cost_feasibility.json',cost)
    refusal=None
    try:assert_training_ready(root)
    except ValueError as error:refusal=str(error)
    assert refusal is not None
    write(root/'training_gate.json',dict(status='REFUSED_BEFORE_MODEL_OPTIMIZER',reason=refusal,
        adapter_artifact=None,trained_steps=0))
    write(root/'crossframe_reuse_summary.json',dict(status='FORMAL_NOT_RUN',
        bridge_object_identity_correct_continuations=bridge['identity_edges'].get('correct',0),
        enriched_memory_extensions={k:v['extensions'] for k,v in semantic['finite_K_playback'].items()},
        natural_memory_extensions=cost['natural_extensions_by_K'],
        natural_exact_visual_prefix_tokens=0,natural_scheduler_admitted_visual_tokens=0,
        formal_PrefillExecutor_skipped_visual_tokens=None,
        formal_matched_tokens=None,formal_reused_tokens=None,
        controlled_cost_PrefillExecutor_skipped_tokens=sum(r['controlled_skipped_tokens'] for r in controls+text_controls),
        controlled_cost_rows=len(controls)+len(text_controls),
        controlled_scheduler_admitted_tokens=sum(r['actual_scheduler_admitted_tokens'] for r in controls+text_controls),
        scope='Controlled skips include fixed text and are not service/world contribution. Natural memory remains absent; formal serving never ran.'))
    write(root/'deployment_decision.json',dict(decision='NO_GO',deploy=False,
        bridge_gate='PASS_POOLED_REGION_GEOMETRY',semantic_gate='PASS_PRETRAINING_FEATURE_PROBES',
        cost_gate='NO_GO',training='NOT_RUN',core_valid_quality='NOT_RUN',
        official_quality='NOT_RUN',ExactKV_service='NOT_RUN',E2E='NOT_RUN',
        formal_config_changed=False,original_weights_modified=False,M1_M2_algorithms_modified=False,
        additional_formal_executors=0,adapter_trained=False,
        failure_axes=['Real supported prefix length/coverage','Added-input cost against complete native visual preservation'],
        next_step='Study a qualified few-region tokenizer/compressor with an explicit current-visual information budget. Do not train this complete513 append candidate, relax identity gates, or claim class-only soft-token gains.',
        limitations=['Five positive bridge scenes, pooled 95% box support is not foreground ownership or broad calibration.',
            'Natural24/three logs is a bounded independent playback, not deployment population coverage.',
            'Real raw feature content is useful in simple probes; LLM usefulness and dynamic-state quality remain unknown.']))
    text=lambda name,content:(root/name).write_text(content.strip()+'\n')
    text('visual_feature_bridge_audit.md',f'''# 真实视觉桥接审计

结论：投影区域接口的有界预训练检查通过；不存在 WorldAnchor 与原生 VLM token 的一一对应。正式部署资格未获认证。

六相机经实际输入 metadata 顺序排列，RGB归一化、resize/crop与K同步更新，本轮处理后输入640×640。EVA/M2输出 `[1,6,1024,40,40]`，原生patch stride16，camera-major、随后row-major。M2复用的是前部窗口及近似全局残差，其后仍有当前视觉计算；不能把窗口复用当作最终LLM输入相等。

`petr3d.extract_img_feat`将M2输出整理为上述网格，Object/Map Head先投影1024→256并结合当前相机几何与历史时序状态。`streampetr_head.py:533–562`把256额外学习查询放在检测查询之前，额外query的reference/query_pos置零，检测框来自去掉额外查询后的分支。最终256个额外输出投影至4096，再追加1个CANbus/ego token。`petr_head_map.py:404–430`同样把256额外查询与1800个lane查询分开，lane由control points表达，额外map输出不构成道路栅格。Native视觉顺序固定为object256、ego1、map256，即 `[1,513,4096]`，LLM扩展时转其计算dtype。Head输出含当前ego/历史变换和全局上下文，通常跨帧不同。

M1通过`omnidrive_adapter.outputs_from_model`把当前检测的ego坐标gravity center/corners经ego_to_world转换为世界框，按类别/大小/距离合法门加一次全局分配关联。普通关联不等于物理身份认证。`state_manager.py:519–528`将整份backbone/object_carrier/map_carrier来源绑定到多个anchor，`associated_feature_ids`没有token级归属或框索引映射。每个anchor的observation support则保存真实相机时间、投影xyxy区域和RGB描述子；描述子也不是前景mask。

新增`worldstream/m3/visual_feature_bridge.py`只读当前提交的M1对象与已有EVA/M2网格。用已校验的camera-time外参、预处理K重新投影选中框的8角，取投影凸包内原生网格中心；按padding前真实image bounds裁剪，并排除其他当前对象support AABB，选择剩余位置最多的相机。无需新编码器、检测器或tracker。读取的是实际观测区域的上下文化特征，既非独立纯对象特征，也不宣称不含背景/全局影响。其他框排除保守，不构成遮挡/前景证明。

160帧/5个日志的富集诊断产生262个合格身份观测，233个可提取区域；29个因支持不足拒绝。独立离线GT只在提取完成后评分，233/233可评分，222/233={bridge['supported_fraction']:.4%}区域满足“至少80%网格中心落在对应GT 3D框投影凸包”，通过预设95%门；137次相邻身份延续正确。11个几何失败例保存在`bridge_analysis.json`，最差支持率50%。GT框投影包含背景，95%门是区域适用性预检，并非95%真实前景纯度；两个开发日志中一个仅有2个区域且1个失败，不外推单日志保证。

原始网格位置中位数52，K64/128/256分别仅95/34/10个区域有足够位置。独立自然24帧中6个合格对象区域均只有15个位置。旧car Core资格没有自动授予64个真实视觉观测的资格。

生成帧字段表示当前M2输出的读取帧，不表示该张量的每个上游窗口都首次在此帧计算；本轮未额外导出每个EVA输出位置的完整M2计算谱系。内容hash绑定实际读取的张量。所有新增桥接代码保持研究接口，未写入正式detector路径。
''')
    text('world_visual_memory_design.md','''# 单个历史区域视觉记忆

实现：`worldstream/m3/visual_memory.py`，一份VisualMemory与一个VisualMemoryManager。记录身份、类别、相机、读取帧、源几何状态版本、最新有效观测、age、revision、scope、真实特征及内容hash。身份仅作生命周期key，投影层只接收实际1024维特征；同类不同对象可以不同。

本轮选择每个memory token对应一个不同原生网格观测，K64/128/256只作有限预算诊断；少于K个位置拒绝，不做补零、复制或把类别复制成K个token。多于K时按原始row-major位置等距下采样成K个。该选择是本轮最小路线的约束，不是所有可学习resampler的信息论下界。

当前帧首次满足身份与区域条件才创建，后续每帧同时重验当前身份资格和区域支持。保持直接返回相同feature tensor与相同projected embedding tensor，未重新生成近似前缀。新对象、类变化、缺失支持、超过8帧、时间断裂、state回滚及scene/epoch/model/adapter/memory/qualification版本变化撤销或重建；revision防止偶然相同内容继承已撤销的视觉KV。修改保存tensor会fail closed。

历史EVA区域仍含旧的外观、位置和背景。本实现承诺其为固定历史上下文，不承诺当前外观不变。位置/速度/朝向/灯态/新对象必须由完整当前513个Native tokens提供，不从旧框或velocity生成稳定语义。语义/运动质量尚未通过训练后LLM测试，不能据此部署。

唯一具体候选布局为 `[P, historical K, current native513, Q]`，无memory直接返回原Native tensor，指针/dtype/顺序不变。投影为单个1024→4096 Linear，测试参数未训练，仅用于实际张量保持和计算成本；不得加载为正式adapter。成本门NO_GO后未选主训练K、未压缩当前视觉、未扩展M3Runtime正式布局。

成本控制直接调用既有PrefixKVCache/PrefixMatcher/ReusePlanner/PrefillExecutor；没有新执行器。源KV由真实LLM forward产生，严格比较实际embedding/position/mask与live执行签名，stable_anchor元数据绑定合法memory scope与revision，动态suffix仍实际prefill。控制实验的显式reuse_length独立于scheduler准入；正式链路集成与scheduler成本表均未获准发布。撤销边界沿用既有matcher裁掉视觉前缀，单测覆盖偶然同值重建。
''')
    text('memory_semantic_analysis.md',f'''# 真实特征语义与保持

桥接门通过后，原始区域特征进行预注册的简单诊断，不输出随机未训练前缀的语言答案。233个合格区域有233个不同实际特征hash。同一身份1秒内观测cosine均值0.9684，不同身份同帧均值0.8540；这并不意味着天然Embedding严格不变。

三个TRAIN日志建立cosine类别原型，两个日志独立开发场景测试；只使用离线GT能确认类别/身份的检测样本，条件宏平均准确率{semantic['category_macro_accuracy']:.4%}，四类均匀机会水平25%。这是一项“给定检测类别正确”的特征信息探针，不是新的检测准确率。相邻帧多候选对象检索{semantic['retrieval_accuracy']:.3%}，按候选数计算的随机水平{semantic['retrieval_chance']:.3%}。离线标签只用于探针/评分，从未传入在线记忆接口。

K64真实富集播放有32次严格同tensor保持；K128为18次，K256为8次。自然24帧在三个K下均为0，详见memory_lifecycle.json与natural_audit/memory_lifecycle.json。历史保持是策略产生的真实相等，不是当前网络特征天然相等。

类别、对象差异证据支持继续成本诊断；车辆颜色/材质等稳定外观属性没有独立标注和测试，不作保证。EVA含位置/背景/当前外观，不能称作状态不变编码器。保留完整当前513以提供当前动态信息，但其能否阻止LLM误用旧上下文仍需训练后空间/运动质量验收，本轮因成本门失败未运行。
''')
    text('composer_training_report.md','''# 训练准入

NOT_RUN。真实特征桥接与预训练内容门通过，成本门NO_GO。`assert_training_ready`在创建训练模型/optimizer前拒绝；trained_steps=0，无WRVMP训练adapter。

上一轮424帧/1272QA训练manifest保持不变，SHA见experiment_plan.json。本轮160帧只补必要区域特征，旧Core资格未自动授予视觉记忆资格。未重新进行完整语言解码。controlled_cost中的untrained_cost_projection.pt只保存成本脚手架参数，trained=false，不能作为训练交付或正式质量输入。冻结原始模型与M1/M2，不重建环境。
''')
    text('core_valid_quality_report.md','''# 真正有Memory的LLM质量

NOT_RUN：成本门失败，未训练WRVMP投影、未运行Core-Valid Native/WRVMP-ZeroKV空间/运动/数量/道路/灯态/新对象问答对比。所有这些任务精度与关键错误数量为null，不用特征分类探针代替LLM质量，不用No-Core多数请求提高质量平均。

No-memory张量直通、保留current513以及撤销边界通过代码单测；没有新真实LLM问答质量试验。有限CUDA控制仅max_new_tokens=1、零decode forward，不进行语言质量归因。
''')
    text('quality_report.md','''# 正式独立质量

NOT_RUN：桥接/内容通过，成本失败使后续训练与六个保留VAL日志正式质量门不具备前置条件。原正式配置与四项绝对下降预算0.10保持不变。

| 指标 | Native A | WRVMP ZeroKV B | WRVMP ExactKV C |
|---|---|---|---|
| mAP | null | null | null |
| NDS | null | null | null |
| DET_l | null | null | null |
| Token F1 | null | null | null |
| 空间/运动严重错误 | null | null | null |

没有未解决严重错误为0的结论；关键错误审查未运行。No-Core原样返回输入是结构保证，不代替完整精度验收。
''')
    table='\n'.join(f"| {k} | {v['by_kind']['native']['executor_wall_ms']['mean']:.3f} | {v['by_kind']['memory_zero']['executor_wall_ms']['mean']:.3f} | {v['by_kind']['memory_exact']['executor_wall_ms']['mean']:.3f} | {v['K_only_prefill_cuda_saved']['mean']:.3f} | 0 |" for k,v in cost_table.items())
    text('performance_report.md',f'''# 成本与性能结论

NO_GO。以下486条计时是原始冻结LLM与既有PrefillExecutor的有限prefix硬件控制；两阶段各243条、原始三个问题、9次计时/问题/K、3轮交替。max_new_tokens=1，真实decode forward数为0；因此不是固定长输出的正式E2E实验，不报告QPS/FPS或整模型speedup。

| K | Native请求wall ms | Memory ZeroKV wall ms | Memory ExactKV wall ms | K本身边际prefill CUDA节省 ms | 自然延续 |
|---|---:|---:|---:|---:|---:|
{table}

K本身节省来自第二阶段相同memory输入的“仅P复用”减“P+K复用”，不把固定文本节省算世界贡献。ZeroKV和ExactKV间的差值是新增布局内局部KV成功；每个K下ExactKV仍比Native慢。追加历史记忆保留全部当前513，命中时仍要计算完整当前视觉并关注新增历史keys，创建帧还增加K个prefill；不能通过跳过K就断言加速。空间预算表同时列出固定513会移除K个当前tokens，未伪造其保真或训练后速度。

自然24帧/3日志保留旧的6个car Core帧/5次延续，但各区域只有15个原生位置，三个K均无memory。相对保持相同通用缓存行为的基线，观察样本内世界增量节省为0，桥接＋实际无记忆update平均{np.mean(overhead):.4f}ms/帧，单问乐观增量净收益{-np.mean(overhead):.4f}ms/请求；三问按帧开销摊销为{-np.mean(overhead)/3:.4f}ms/请求。这不是完整A/B/C服务延迟，未量测项乐观置0，仍不支持训练。场景重采样区间与局限见prefix_cost_feasibility.json；正式配对E2E CI为null。

实时正样本桥接wall均值1.8490ms/帧（其中M1读取/资格0.5393ms，提取/同步1.3098ms；正样本富集分布）；CUDA stream event跨度均值1.8345ms包含CPU投影/launch间隙，不能当成纯GPU kernel时间。该跨度提供本路径GPU活动的上界，纯桥接kernel累计时间为null。独立真实memory创建/保持/投影的CUDA event与GPU profiler活动见memory_transaction_cost.json，projection+组装活动见controlled_cost/gpu_activity_probes.json；分开计量，不能把profiler活动和wall重复相加。

M1/M2/特征矩阵已经计算，源captured KV的初始化不在目标hit wall内；目标executor的branch clone、当前KV更新/ownership/capture和同步在wall内。显式match/planner另计，live执行签名构建在overhead.json记录。真实memory creation包括clone/finite/hash D2H，未把它藏在warm投影成本中。正式服务未运行，所有实际GPU峰值仅对硬件控制有效，见summary.json。三问通用同帧缓存收益未重做，未记作WRVMP世界贡献。
''')
    text('result_answers.md',f'''# 十四个问题的证据回答

1. WorldAnchor对应当前检测/世界几何与连续关联；VLM object256/map256是额外全局学习查询，整组associated_feature_ids不提供token↔对象映射。
2. 可以建立有界可信的相机投影区域↔EVA/M2网格接口：233区域、222通过框支持、137正确身份延续；不升级为纯前景/对象专属embedding，11失败和29拒绝保留。
3. 是，原始记忆特征来自真实1024维视觉网格且每个区域hash不同；并非只含car字符串。LLM可用性尚未证明。
4. 同类不同对象可不同。历史特征与投影tensor直接持有同一实例；K64富集32次严格保持。当前特征本身并不严格相等。
5. 只把旧区域当历史上下文，当前513全部保留；不把旧pose/velocity/灯态声明为当前事实。是否造成LLM误用历史，质量NOT_RUN。
6. 富集数据每区域4–423个不同网格位置，中位52。自然合格对象只有15个位置；本轮K64/128/256路线可创建/保持的真实视觉prefix为0。15→64学习重采样未实现，不能把它当64个独立观测。
7. 不能。自然观察内世界节省0，单问新增开销使乐观增量净收益约{-np.mean(overhead):.3f}ms/请求，未明显优于旧+0.084ms。样本外总体净收益未知。
8. 富集桥接平均wall1.849ms，其中读取/资格0.539ms、提取同步1.310ms；自然成本和memory各阶段见prefix_cost_feasibility.json。桥接CUDA stream1.834ms含CPU间隙，纯kernel未知；真实创建/投影GPU活动已单独profile。
9. 未训练，成本门在创建optimizer前拒绝；cost projection是untrained脚手架，无正式adapter。
10. 未证明，真正有memory的空间/运动问答NOT_RUN，不能用原始特征90.1%类别探针代替。
11. 正式Prefill跳过NOT_RUN，正式世界视觉服务tokens=0；有限控制确实跳过P+K，另记controlled计数，scheduler准入为0，不把对象关联乘K冒充服务复用。
12. 有局部硬件控制ZeroKV→ExactKV收益；训练后/自然服务B→C收益NOT_RUN。
13. 没有已验证的完整E2E加速；完整当前513＋memory即使控制命中也慢于Native。正式A/B/C、decode CUDA、QPS/FPS和scene配对CI均NOT_RUN/null。
14. 本轮主要失败于可支持prefix粒度/自然覆盖和完整视觉保留时的新增输入成本。真实视觉关联与特征内容已有正证据；对象纯前景、稳定外观和LLM输入适配仍未证明。该NO_GO仅针对所实现最小路线，不声称所有少区域resampler/有损视觉压缩必然失败。
''')
    sources=[p for folder in ('worldstream/m3','tools/worldstream_m123') for p in Path(folder).glob('*visual_memory*.py')]
    sources += [Path('worldstream/m3/visual_feature_bridge.py')]
    write(root/'source_and_artifact_bindings.json',dict(
        base_commit='25d782296ff119fad945c1f9a029689a0ded3389',
        base_weights_sha256='6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d',
        adapter_trained=None,untrained_cost_projection_sha256=sha256(root/'controlled_cost/untrained_cost_projection.pt'),
        source_sha256={str(p):sha256(p) for p in sources},
        raw_files_sha256={str(p):sha256(p) for folder in ('bridge_capture','natural_audit/bridge_capture')
            for p in (root/folder).glob('*.pt')},
        note='Raw tensors stay local/gitignored with hashes. bridge_capture/executed_capture_source.py and controlled_cost/executed_cost_source.py preserve earlier stage source snapshots. Final post-capture validation-only/reproduction options do not change reported forward inputs. Memory timestamp validation added after initial semantics/cost launch; CPU semantic playback rerun with final code.'))
    print('WRVMP_REPORT NO_GO; training refused; formal configuration retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/m3_world_visual_memory_run01'))
    generate(p.parse_args().root)
