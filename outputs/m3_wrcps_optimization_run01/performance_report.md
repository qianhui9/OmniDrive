# 性能验收

状态：NOT_MEASURED_STAGE_ONE_FAILED。真实世界状态门槛失败，没有生成合法的已训练WRCPS B/C请求，无法计算相对Native或ZeroKV的加速比，也无法计算场景级配对95% CI。

|指标|WRCPS实测|
|---|---|
|Prefill CUDA / Decode CUDA|未测|
|M3 Wall / Model Wall / Request Wall|未测|
|Composer Wall|未测|
|完整在线Anchor开销|未测|
|GPU峰值显存 / Cache显存 / 吞吐量|未测|
|C相对B完整请求净收益|未测|
|Native/C E2E Speedup与场景配对95% CI|不适用|

唯一新增模块计时是对48个真实不可变M1快照进行一次CPU重放：Anchor管理平均 0.016449 ms/帧。该值包含快照验证及Anchor更新，不包含六相机输入、M1状态导出、Composer或GPU执行，不能当作完整在线开销或系统加速证据。重放不增加独立场景数。

`anchor_audit_language`是真实六相机原生布局M123可行性审计，不属于WRCPS A/B/C性能实验。原始时延保存在该目录，但不用于计算WRCPS收益。`validation_plan.json`保留完整请求测量、原序1/3问题、原生相同参考tokens/Decode步数、三轮交替、6个以上独立场景和scene bootstrap的验收设计，全部明确标为未运行。

小模型测试跳过128个严格相同的合成稳定视觉token，只是执行合同验证，不是生产模型Prefill CUDA或E2E收益证据。
