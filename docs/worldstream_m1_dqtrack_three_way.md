# M1 与 DQTrack 三组离线配对实验

生产默认仍为 `dqtrack_mode=off`，管理器继续拒绝退役 Hint。只有 `tools/worldstream_m1/dqtrack_three_way_policy.py` 中显式构造的离线子类使用外部身份；生产 `_associate()` 仅等价转发既有 `associate()`。

A 使用 `M1Config()` 的实际完整默认配置与普通管理器。B 在当前合法延续上增加成熟外部身份冲突拒绝，缺失证据保留原行为。C 额外要求整个合法竞争分量存在完整且唯一的外部身份映射，否则回退 B。地图匹配不变。三组保持同一几何门限、预测、状态管理和评分配置。

固定研究条件：tracking_score≥0.5、连续至少3次已提交对齐、年龄≤1秒；这些是复用历史研究 profile 的未校准过滤条件，不是正确概率，也不与距离加权。来源帧、状态版本、epoch、执行签名和来源必须一致。元数据只在原提交事务的私有 fork 上更新；异常不发布。外部 ID 不签发物理证据、不提升速度或 M2 资格。当前桥接要求跟踪器与检测在两侧均没有其他合法对齐，因而严格限制 C 的可用消歧机会。

命令中的输出目录必须尚不存在：

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 -q \
 > /tmp/m1_three_way_tests.log 2>&1

taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.dqtrack_three_way_eval \
 --output outputs/NEW_UNIQUE_DIRECTORY --test-log /tmp/m1_three_way_tests.log
```

入口校验原协议 SHA256、候选与选择文件 provenance、缓存帧/场景/时间映射和原跟踪导出 run/checkpoint 声明；冻结完整配置、源文件快照、图像/标注/候选/轨迹哈希后回放。三组逐场景独立初始化，按场景轮换执行顺序，不运行模型、不运行外部跟踪器。缓存原始来源标记为 `historical_full_model_detection_replay`，不改写成新推理。

GT 仅在 commit 后标注：类别、尺寸比≤2、传感器支持、中心≤1米且双向唯一。unknown 不进入 scored。effective recall 的分母是相邻帧共同可评分的候选实例；coverage 分母排除每个场景首帧。legacy fragments 统计可评分 GT 观测之间的帧间断，与官方 FRAG 不同。默认历史开关为关闭，不用历史复杂主方法数字替代新 A。计时是 CPU 缓存回放 precheck+候选深拷贝+commit，包含桥接文件首次解析，排除图像加载、GT 评分和 DQTrack 模型推理，不能解释为联合服务吞吐。

本次实际产物见 `outputs/m1_dqtrack_three_way_20261021_run02/`。run01 只留下了过严来源类型预检的失败记录，未执行任何组；不存在调参或重复方法回放。
