# Composer 训练与数据准入

训练命令实际调用了现有 wrcps_train.py --spatial --stable-tokens 64 --rank 128 --epochs 3，但 validate_manifest 在加载基础模型、构建优化器之前拒绝。不是训练完成，也不是训练OOM。实际优化器更新0，没有SF checkpoint、最佳epoch或训练loss；旧K128权重不作为新结构权重使用。原始traceback见 adapter_training.log、adaptation_driver.log，返回码见 commands.json。

真实开发采集10个新TRAIN Log、80帧、240个原始QA，划分6训练Log/48帧、4验证Log/32帧。训练集有效核心帧 1/48、跨帧核心延续 0/42；开发验证有效核心帧 6/32、跨帧延续 5/28。逐场景和生命周期见 training_admission.json。大量真实无核心帧不能复制或伪装成有效核心序列。

独立资格VAL的5次正确命中满足最初适用性门槛，但不足以保证另选开发Log中也有持续核心。本轮在固定开发采集后观察到这一失败，没有转用资格或最终验收Log补训练，也没有按输出搜索新场景、放宽car规则或增加epoch。

实现已支持原始Answer KL、Task CE与固定0.5方向词KL。方向增强只在参考方向词位置且Teacher argmax同意参考时启用，冲突单列；基础权重冻结、梯度检查点沿用。无核心为原生identity，不做虚假更新；checkpoint选择只看有效核心开发验证loss。配置见 composer_training_config.json。以上为实现和合同测试，不是本轮实际学习结果。

K32只验证接口/布局合同，没有开发质量证据触发额外训练。峰值训练显存未测量（null），不能用旧训练显存代填。
