# 15项问题回答

1. **为什么过去48训练帧只有1个有效Core？**

47帧无合格事实；car首失败为{'no_current_consecutive_observation': 181, 'insufficient_consecutive_observations': 52, 'class_score_below_calibrated_range': 86, 'association_residual_above_calibrated_range': 1}。分数/连续观测及ID当前支持限制占主，残差是更后级过滤；不能将失观测全部解释为物理遮挡。

2. **扩大窗口提高多少？**

原6训练Log：8/16/32帧窗口有效帧1/2/3，延续始终0。全部14训练Log有效帧45/93/195、延续28/53/108；开发验证6/13/22有效帧、5/9/11延续。

3. **更多合法TRAIN Log有足够正样本吗？**

是，数据门GO。14训练Log中8个正Log，87个有效Anchor、39条真正持续序列，108次已评分正确延续，0错误/未知；保护资格/开发验证/验收Log不进训练。历史暴露已披露。

4. **富集前后规模？**

训练原始候选448帧、195有效/108延续；富集296帧、195有效/108延续。有效帧占比43.53%→65.88%，不能当作部署覆盖率。验证完整128帧、22有效/11延续。

5. **car Prefix有语言语义价值吗？**

所有有效单car事实的内容编码相同，ID不进embedding；有效性条件可能有价值，但尚无训练后增量语言效用证据。三问有限反事实证明能影响logits/回答，不证明影响有用。

6. **K64潜在信息损失？**

以64稳定槽替换64原生视觉槽，object/map各移除32并修改部分合并目标。Ego/未混合槽保留，但空间/运动/数量依赖仍可能丢失；原生latent索引不是物理邻接。

7. **真实覆盖下prefill节省？**

旧独立VAL Anchor5/21转换；按请求和layout回放，1问Stable64机会5/24，模型预期prefill节省2.9433ms/request。CUDA上限与服务结果分列。

8. **扣除额外成本有净收益吗？**

单问乐观总净值0.0839ms/request，世界前缀增量2.5663ms/request；相应95%计时区间{'total_net_ms': [-0.9341879118662182, 1.2786682933909435], 'marginal_world_net_ms': [2.502708368267095, 2.633821461144918], 'scope': 'Nine micro repetitions, questions grouped within repetition; Composer resampled separately. Measurement uncertainty only, not scene/population E2E CI.'}。收益门No-Go。未测额外成本置0，已偏向方法。

9. **完成训练了吗？**

否。现有训练预检已通过，但收益门未通过，因此优化器更新0，新checkpoint/loss/训练显存null；不是再次数据拒绝训练。

10. **有效核心关键质量达标吗？**

尚无已训练SF证据，NOT_RUN。未训练的反事实、无核心identity及旧回归均不能代替验收。

11. **新独立场景命中率和寿命？**

本轮独立于训练Log的开发验证11/124转换（8.87%），有效Anchor均值2帧，2/4 Log有延续；这些是既有开发验证Log扩窗，非完全未见最终VAL。旧独立资格VAL5/21、单Anchor6帧用于收益机会估计。最终保留场景未测。

12. **真实Exact Visual KV tokens多少？**

正式服务0；micro控制执行确实按原Executor跳过了合法prefix，但所有控制tokens从正式服务统计中排除。

13. **B→C真实Prefill/M3 Wall改善？**

NOT_RUN/null。只有控制性全prefill/text/text+64及同帧prefix计时，不是已训练B/C因果服务对照。

14. **A→C可重复E2E净加速？**

未证明。未执行完整自然tokens、相同decode步数、三轮正式交替和场景配对95%CI。多问同帧控制收益属于通用同帧复用，不能冒充跨帧世界收益。

15. **当前路线是否值得继续？**

数据阻塞已解决，工程实现保留；在当前单car内容、K64预算及真实稀疏覆盖下，收益前置门不足，建议暂停高成本训练和部署替换。该判断限于当前结构/硬件/覆盖证据，不是宣称所有世界状态KV方法不可能有效。
