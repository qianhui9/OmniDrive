# WRSAR v2 delivery

Decision: **PARTIAL_GO**, limited to current independently scored facts. Broad driving quality remains INCOMPLETE.

Implemented opt-in SHA256 question RNG isolation around the existing verifier/native generation; compact current binding and historical unobserved-competitor risk inside the existing WRSAR proposal; no second Runtime/tracker/neural network. Historical centers are transformed through actual Ego poses and used only for risk. Every fast fact remains current observed perception. Misses use existing K8 target-aware generation. All research options default off; no weights, training, generation parameters or M1/M2 algorithms changed.

Formal TEST uses three new logs, twelve current frames, separate natural/fixed/addressable and 1Q/3Q workloads. Addressable3Q B→C 1.3146x, scored paired loss -0.216667; 45 actual M3 skips. Newly severe outputs versus B: 0. Cross-frame contribution is explicitly measured with the same policy and RNG in temporal_state_contribution.json; no independent benefit is claimed without measured evidence.

## Delivery interpretation

5. 新机制对已观测竞争、边界和历史潜在遗漏做风险判断；DEV及本轮 TEST 已评分快速回答没有错误。但旧 Car/Truck 案例仍未解决。它是该流首帧，没有合法历史；当前绑定和时序机制都仍会输出Truck。不能根据零个新 TEST 命中错误就声称修复了所有漏检。

7. 尚未证明独立时序收益。三问 Current-Only 命中51/108，Temporal-Aware命中42/108；已评分质量均为80.00%。少掉的9次命中都没有可靠当前标签。C-current→C-temporal实测比值0.9276，时延差CI为[-34.53695036781331, 305.856308589379]ms，跨零。当前检测逐项一致，故可确认时序项改变了决策，但无法确认改善了质量或效率。

15. 相对B新增可证实严重错误为0。C仍有9条严重输出（含重复），均在正常目标模型生成路径；没有已评分快速回答严重错误。旧诊断B/C同错也保留，不能把“没有新增”写成“系统没有错误”。
