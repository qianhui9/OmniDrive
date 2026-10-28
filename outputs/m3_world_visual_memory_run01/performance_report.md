# 成本与性能结论

NO_GO。以下486条计时是原始冻结LLM与既有PrefillExecutor的有限prefix硬件控制；两阶段各243条、原始三个问题、9次计时/问题/K、3轮交替。max_new_tokens=1，真实decode forward数为0；因此不是固定长输出的正式E2E实验，不报告QPS/FPS或整模型speedup。

| K | Native请求wall ms | Memory ZeroKV wall ms | Memory ExactKV wall ms | K本身边际prefill CUDA节省 ms | 自然延续 |
|---|---:|---:|---:|---:|---:|
| 64 | 178.281 | 219.726 | 184.597 | 2.768 | 0 |
| 128 | 180.801 | 233.329 | 188.450 | 41.467 | 0 |
| 256 | 181.182 | 275.433 | 192.916 | 82.869 | 0 |

K本身节省来自第二阶段相同memory输入的“仅P复用”减“P+K复用”，不把固定文本节省算世界贡献。ZeroKV和ExactKV间的差值是新增布局内局部KV成功；每个K下ExactKV仍比Native慢。追加历史记忆保留全部当前513，命中时仍要计算完整当前视觉并关注新增历史keys，创建帧还增加K个prefill；不能通过跳过K就断言加速。空间预算表同时列出固定513会移除K个当前tokens，未伪造其保真或训练后速度。

自然24帧/3日志保留旧的6个car Core帧/5次延续，但各区域只有15个原生位置，三个K均无memory。相对保持相同通用缓存行为的基线，观察样本内世界增量节省为0，桥接＋实际无记忆update平均0.5699ms/帧，单问乐观增量净收益-0.5699ms/请求；三问按帧开销摊销为-0.1900ms/请求。这不是完整A/B/C服务延迟，未量测项乐观置0，仍不支持训练。场景重采样区间与局限见prefix_cost_feasibility.json；正式配对E2E CI为null。

实时正样本桥接wall均值1.8490ms/帧（其中M1读取/资格0.5393ms，提取/同步1.3098ms；正样本富集分布）；CUDA stream event跨度均值1.8345ms包含CPU投影/launch间隙，不能当成纯GPU kernel时间。该跨度提供本路径GPU活动的上界，纯桥接kernel累计时间为null。独立真实memory创建/保持/投影的CUDA event与GPU profiler活动见memory_transaction_cost.json，projection+组装活动见controlled_cost/gpu_activity_probes.json；分开计量，不能把profiler活动和wall重复相加。

M1/M2/特征矩阵已经计算，源captured KV的初始化不在目标hit wall内；目标executor的branch clone、当前KV更新/ownership/capture和同步在wall内。显式match/planner另计，live执行签名构建在overhead.json记录。真实memory creation包括clone/finite/hash D2H，未把它藏在warm投影成本中。正式服务未运行，所有实际GPU峰值仅对硬件控制有效，见summary.json。三问通用同帧缓存收益未重做，未记作WRVMP世界贡献。
