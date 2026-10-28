# 十四个问题的证据回答

1. WorldAnchor对应当前检测/世界几何与连续关联；VLM object256/map256是额外全局学习查询，整组associated_feature_ids不提供token↔对象映射。
2. 可以建立有界可信的相机投影区域↔EVA/M2网格接口：233区域、222通过框支持、137正确身份延续；不升级为纯前景/对象专属embedding，11失败和29拒绝保留。
3. 是，原始记忆特征来自真实1024维视觉网格且每个区域hash不同；并非只含car字符串。LLM可用性尚未证明。
4. 同类不同对象可不同。历史特征与投影tensor直接持有同一实例；K64富集32次严格保持。当前特征本身并不严格相等。
5. 只把旧区域当历史上下文，当前513全部保留；不把旧pose/velocity/灯态声明为当前事实。是否造成LLM误用历史，质量NOT_RUN。
6. 富集数据每区域4–423个不同网格位置，中位52。自然合格对象只有15个位置；本轮K64/128/256路线可创建/保持的真实视觉prefix为0。15→64学习重采样未实现，不能把它当64个独立观测。
7. 不能。自然观察内世界节省0，单问新增开销使乐观增量净收益约-0.570ms/请求，未明显优于旧+0.084ms。样本外总体净收益未知。
8. 富集桥接平均wall1.849ms，其中读取/资格0.539ms、提取同步1.310ms；自然成本和memory各阶段见prefix_cost_feasibility.json。桥接CUDA stream1.834ms含CPU间隙，纯kernel未知；真实创建/投影GPU活动已单独profile。
9. 未训练，成本门在创建optimizer前拒绝；cost projection是untrained脚手架，无正式adapter。
10. 未证明，真正有memory的空间/运动问答NOT_RUN，不能用原始特征90.1%类别探针代替。
11. 正式Prefill跳过NOT_RUN，正式世界视觉服务tokens=0；有限控制确实跳过P+K，另记controlled计数，scheduler准入为0，不把对象关联乘K冒充服务复用。
12. 有局部硬件控制ZeroKV→ExactKV收益；训练后/自然服务B→C收益NOT_RUN。
13. 没有已验证的完整E2E加速；完整当前513＋memory即使控制命中也慢于Native。正式A/B/C、decode CUDA、QPS/FPS和scene配对CI均NOT_RUN/null。
14. 本轮主要失败于可支持prefix粒度/自然覆盖和完整视觉保留时的新增输入成本。真实视觉关联与特征内容已有正证据；对象纯前景、稳定外观和LLM输入适配仍未证明。该NO_GO仅针对所实现最小路线，不声称所有少区域resampler/有损视觉压缩必然失败。
