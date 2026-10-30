# 冻结主实验复现

当前唯一活动路线为 R0：M1 阈值 0.3、M2 budget 3、M3 current_only / C-basic / .45 / .75。配置、GPU 实测收据及报告均在本目录。生产默认关闭；残余严重错误与独立 TEST 不完整结论不变。

在项目根目录执行完整保留校验（不进行 GPU 推理）：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.freeze_cleanup verify
```

只重放冻结主路线，不重新搜索或选择参数：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.freeze_cleanup prepare-replay --output /tmp/omnidrive-r0-main-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_capture main --root /tmp/omnidrive-r0-main-replay
```

第二条需要空闲 GPU 并持有既有锁；GPU 被占用只等待，不终止其他进程。新目录必须不存在。A/B/C 五种负载共 240 次新执行；原始主报告是 208 次新执行加 32 个历史收据复用，不能合并均值。重放既有 DEV 不构成独立 TEST。

离线报告重组建议先复制本目录到 /tmp，再对副本运行 final_quality_analysis report --root <副本路径>，以保留冻结输出字节。读取器直接使用 requests.jsonl.gz，无需恢复大型 Tensor。

provenance/parent_evidence 只包含本主实验实际依赖的历史对照收据及小型报告；resources 是原运行资源的逐字节副本。consolidation/plan.json 记录删除的旧输出及原始 Git 提交。历史完整性校验将已删除文件与该提交的 Git blob 比较，将执行源文件与 original_sources 中原始快照比较，明确区分当前文件、历史源码和 Git 归档；不宣称旧目录仍然存在。

原始完整历史复现说明保存在 consolidation/original_docs/REPRODUCE.md，仅为来源记录。恢复整个旧研究流程须在旧标签 m123-main-r0-frozen-20261030 对应的独立 checkout 中进行，不向当前 outputs 恢复旧目录。全部旧审计与 NO_GO 结论可从该标签读取。

权重 ckpts/iter_10548.pth 的 SHA256 为 6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d；权重、数据集软链接、Conda 环境和 Git 历史保留。本轮仅修改资源路径与离线保留校验，不重新运行 GPU 实验。
