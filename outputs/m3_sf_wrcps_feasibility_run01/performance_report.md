# CUDA潜在收益与正式效率边界

本轮完成真实原生LLM和原PrefillExecutor的有限CUDA控制：3个原始问题、2轮预热、9轮有效测量、5种交替控制，共135条有效计时。完整prefill与text/text+64/same-frame-prefix的embedding/position/mask严格一致；源来自真实核心延续相邻帧。控制性执行器实际suffix长度逐次断言，跳过量与forward长度严格相符。每次只生成1token/零decode-forward是micro定义，不是缩短自然服务来制造加速。另有5个Kineto GPU活动探针，独立于Wall统计。

|控制|模型Prefill CUDA stream ms|完整控制调用Wall ms|Planner Wall ms|
|---|---:|---:|---:|
|native|178.2525|179.2763|0.3818|
|full|179.1192|181.4352|0.3946|
|text|178.2920|181.3643|0.3849|
|text_plus64|164.9914|168.0357|0.3826|
|same_frame_visual|29.9992|32.4849|0.3727|

CUDA列是包围真实model forward的event stream区间，包含host提交空隙，不冒充纯kernel时间。GPU活动时长和事件数单列于 controlled_micro/gpu_activity_probes.json；CPU控制与完整Wall单列。Cache Branch、owned KV clone及cache更新在既有Executor的控制调用内计时，Native真实生成作为共同开销参照。

Stable64相比仅固定文本的模型CUDA边际节省均值 13.3006ms/hit；完整prefill减text+64为 14.1278ms/hit。K64 Composer真实结构warm Wall均值0.7377ms/core-valid帧；同帧3问只构造一次，已按请求摊分，cold/slot matching及GPU区间在composer_cost.json。成本使用未训练参数，只作结构/hardware测量，不作质量或正式SF加速证据。

每帧1问：Stable64机会5/24，预期纯prefill节省2.9433ms/request；总净值估计0.0839ms、计时区间[-0.9341879118662182, 1.2786682933909435]；扣除Composer/状态成本后的增量世界前缀净值2.5663ms、区间[2.502708368267095, 2.633821461144918]。
每帧3问：Stable64机会5/72，预期纯prefill节省0.9811ms/request；总净值估计97.6338ms、计时区间[96.85411677077084, 98.52401732886064]；扣除Composer/状态成本后的增量世界前缀净值0.8554ms、区间[0.8342361227556985, 0.8779404870483061]。

覆盖依据是旧独立资格VAL的3Log/24帧真实生命周期，而非本轮富集训练率。回放按单entry、场景/layout重置、原问题顺序计算机会；同帧收益单列，不能解释成世界跨帧收益。真实现行Scheduler缺少新Adapter/Layout的质量成本绑定，实测准入0；乐观估计只假设未来对真正Exact候选校准准入，不假设Anchor命中100%。

期望总净值以Native Wall减去按机会加权的Executor Wall、Planner、warm Composer及资格选择/Anchor CPU回放成本。完整候选提取/protocol hashing、M3执行signature、SF展开、额外allocator成本未测而乐观置0；新建Anchor也用较低warm Composer成本，因此结果偏向方法。它不是已测完整M3/Request净值。bootstrap仅描述9轮micro计时不确定性，不是场景配对E2E置信区间。

独立资格原始输入长度569–588tokens，本次硬件控制为572/574/575tokens。成本转移仅限这一有限近邻长度假设，不是全局服务成本模型。首次控制脚本解包错误在预热阶段断言退出；第二次完成135条计时后因未训练反事实的非有限派生数字指标写JSON失败。两次失败日志保留，第二次完整计时也保留在earlier_timing_run_sensitivity，未隐藏；最终135条控制与完整开销来自成功重放。规则/问题/参数未因这些结果改变。非有限外部诊断标Unknown，性能计时仍严格要求finite。

收益门：**NO-GO**。按测量前冻结的规则，1问总净值和世界前缀增量净值的95%计时下界必须同时>0；多问同帧通用收益不能替代跨帧世界前缀收益。结论是当前单car/K64路线没有足够稳健的训练投资收益证据，不是数学证明一切硬件/布局都不可能加速。

正式A/B/C自然服务、固定完整参考tokens/相同decode步数、三轮交替及场景级配对CI均NOT_RUN。真实正式Exact Visual KV tokens=0，实际服务跳过prefill tokens=0，正式B→C Prefill/M3 Wall改善、A→C E2E speedup、FPS/QPS、自然Decode CUDA均null。micro控制跳过量绝不填入正式复用。
