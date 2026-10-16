# 数值误差归因

A=原始 EVA；B=前两层全部窗口走原 Packed Attention/MLP，不读取历史结果；
C=前两层闭包稀疏执行，其余保持原始 blocks。两个独立 seed，无阈值放宽。

| Seed | 输入 | B−A MaxAbs | C−A MaxAbs | 原始 Dense 单相机拆批−A MaxAbs |
|---|---|---:|---:|---:|
| 2801 | repeat | 0 | 0 | 0.587139 |
| 2801 | patch | 0 | 0.276685 | 0.587139 |
| 2801 | window | 0 | 0.347174 | 0.587139 |
| 2801 | multi_window | 0 | 0.48756 | 0.587139 |
| 3509 | repeat | 0 | 0 | 0.774061 |
| 3509 | patch | 0 | 0.192083 | 0.771832 |
| 3509 | window | 0 | 0.139725 | 0.771832 |
| 3509 | multi_window | 0 | 0.439499 | 0.771832 |

全部窗口的 B 在本组输入与 A 相同；Sparse 和纯 Dense 拆批都产生差异。
首层 Sparse 差异约 1e-3，后续 Global/deep blocks 放大；逐层位置/数值见 numerical_raw.json。
这支持 batch shape/GEMM/FP16 QKV 舍入是主要来源，不证明所有浮点误差都安全。
原始 norm1 在 padding 前；Packed 用 valid mask 将 padded norm1 保持零；RoPE 与原 attention 对象相同。
原始 MLP 在 unpad 后 40×40 token 上计算，Packed 对补齐的窗口计算；全量 B 无差异不表示所有子批次数值相同。
诊断恢复原始 MLP/unpad 顺序的结果也记录；没有修改原始 EVA Forward、attention scale 或 residual。
固定全量 batch 的 shadow 另计算全部窗口后，仅将闭包证明不变的窗口替换成旧 cache，
最终 MaxAbs 为 0，见 cache_substitution_shadow.json。这将历史替换本身与缩小 batch 的差异分开；
该 shadow 仍执行全部窗口，没有计算节省，也不作为在线优化。

同值输入的逐组件拆分进一步记录如下。QKV 在 FP32 中仅有数个 ULP 差异，也可能跨过 FP16 舍入边界；
attention_output 包括原 Flash/inner norm/output projection，不将整段差异全归给某个未隔离的 kernel。
K/V 同样逐项保留在 component_numerical_audit.json。Norm1 的零差异和固定 batch shadow 的零差异
提供缓存替换与浮点执行形状的区分证据，不把小特征误差当作任务安全。

| Block | 子批窗口 | Norm1 MaxAbs | Q投影 FP32 MaxAbs | Q进入Flash的FP16变化元素数 | Attention输出 MaxAbs | 同输入 MLP有效token MaxAbs |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 5.72205e-06 | 613 | 0.000551701 | 1.00136e-05 |
| 0 | 6 | 0 | 6.19888e-06 | 2846 | 0.000786543 | 1.28746e-05 |
| 0 | 36 | 0 | 0 | 0 | 0.000784039 | 0 |
| 1 | 1 | 0 | 3.8147e-06 | 438 | 0.000601768 | 4.29153e-06 |
| 1 | 6 | 0 | 3.33786e-06 | 2038 | 0.000486493 | 7.39098e-06 |
| 1 | 36 | 0 | 0 | 0 | 0.000763655 | 0 |

Dense camera 拆批的 Patch Embedding MaxAbs 为 0。
实际 matmul TF32 allowed=False，cuDNN TF32 allowed=True，
这些是原始执行策略，未为通过目标而改变；allowed 不代表每个 conv kernel 都使用 TF32。

完整模型 A/B/C 与 camera 子批次的 Detection/Map/native Memory/greedy/planning 对照见 correctness_results.json
和 task_quality_summary.json。MaxAbs、同类检测匹配、Map 各叶与 Memory 联合匹配分别报告，
不能用全局 Relative L2 通过认证。未匹配 Memory 的大差异可涉及 top-k slot 顺序，匹配后残差和时间戳仍保留。
正式 Window Auto 沿用先前批准集合 [0]；没有因为 B 全量相同而批准 C 子批次。
Camera Auto 也只采用实际 table 标明的已验证数量；Exact 模式是显式实验开关。
