# WRVMP 复现与阶段验收

当前目录必须是 `/data/qh/phdCode/work4/OmniDrive`。基线提交 `25d782296ff119fad945c1f9a029689a0ded3389`；最终分支、实现与归档提交见 `delivery.json` 及 Git log。正式配置没有修改。结论是成本门 NO_GO，并非已经训练或部署。

使用既有 `OmniDrive` 环境，不安装包、不重建环境。解释器 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。环境声明沿用 `outputs/m3_wrcps_optimization_run01/environment_spec.json`，SHA256 `eda83d61075d3610f664d5edffd4c5bf070c6dfed0822bf086252977b5c3e2e5`。所有命令避开故障CPU4/5。

## Environment witness (verbatim)

先确认GPU memory.used小于500MiB；若占用，只监测等待，不停止任何其他进程。在Codex沙箱无法访问CUDA时，使用host审批执行相同命令并记录差异，不更改环境。

```bash
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -c 'import os,torch; torch.manual_seed(0); assert not ({4,5}&set(os.sched_getaffinity(0))); x=torch.arange(64,dtype=torch.float32,device="cuda").reshape(8,8); value=float((x@x.T).sum()); assert value==510720.0; print("WITNESS",value,torch.cuda.get_device_name(0)); print("AFFINITY",sorted(os.sched_getaffinity(0)))'
```

预期 `WITNESS 510720.0 NVIDIA GeForce RTX 3090`，affinity不含4/5。fresh-agent记录位于 `doc_validation.json`。

## 重算现有证据（不载入完整模型）

本地原始特征 `.pt` 未提交Git，hash见 `source_and_artifact_bindings.json`；从交付目录复制回来后可重算。下面三条顺序执行，保留报告中各阶段来源区别。

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory_analysis
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory_semantics
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory_report
```

语义重算只用真实特征作简单类别/检索探针与张量生命周期检查，不生成语言答案。报告工具检查当前自然播放仍无memory；如证据变化，会报错要求重审成本模型，不自动批准训练。

## 从六相机与原始权重重新捕获

以下新目录必须不存在，不能覆盖原run01。旧的训练数据清单、原始QA、nuScenes完整相机数据与已有原生carriers仍需可用；没有重新下载或构造替代数据。GPU工具内部均使用 `gpu_job_lock` 和显存占用等待。一次只运行下一条，上一条完成且阶段门通过后再继续。

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory --stage prepare --root outputs/m3_world_visual_memory_reproduce01
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory --stage audit --root outputs/m3_world_visual_memory_reproduce01
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory_analysis --root outputs/m3_world_visual_memory_reproduce01
```

预设桥接门见 experiment_plan.json。仅 bridge_analysis.status=PASS 后继续：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory_semantics --root outputs/m3_world_visual_memory_reproduce01
```

仅 memory_semantic_probe.status=PASS 后继续：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory --stage prepare-natural --root outputs/m3_world_visual_memory_reproduce01
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory --stage audit --root outputs/m3_world_visual_memory_reproduce01/natural_audit
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory_cost --root outputs/m3_world_visual_memory_reproduce01
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory_cost --text-control --root outputs/m3_world_visual_memory_reproduce01
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory_overhead --root outputs/m3_world_visual_memory_reproduce01
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory_report --root outputs/m3_world_visual_memory_reproduce01
```

160帧富集桥接＋24帧自然播放都使用原M1/M2、零LLM问题，不作为服务时间。两阶段成本控制各243条，共486条，K64/128/256、三个原始问题、9次计时/问题/K，3轮交替。首阶段Native/ZeroKV/ExactKV，第二阶段Native/仅P复用/ P+K复用，明确分离K与固定文本。每次仅计算一次prefill、零decode forward；不作语言质量或固定长decode的正式E2E声明。

实际run01自然清单由同一固定筛选规则生成；最终prepare-natural把scene字典顺序规范化，JSON hash可能因早期字典序不同而不同，token顺序/相机数据/日志划分必须相同。早期执行脚本快照保存在bridge_capture/executed_capture_source.py和controlled_cost/executed_cost_source.py；后续增加的时间合法性校验、VAL数据选项、固定文本控制和复现入口都已记录。重算语义播放使用最终代码。

## 回归验证

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python outputs/m3_world_visual_memory_run01/validate_gpu.py
```

查看environment.json、final_test_result.json、tests_gpu_final.log。测试包含现有全部M1/M2/M3/M123及新增投影、padding/camera顺序、真实张量保持、版本撤销、偶然同值KV边界与训练拒绝门。

## 训练和正式验收

本轮成本NO_GO，训练/六日志质量/正式A-B-C/固定decode E2E全部NOT_RUN。没有训练WRVMP adapter，没有修改正式配置、原始权重、M1/M2主算法或现有PrefillExecutor。不得把untrained_cost_projection.pt作为模型交付加载。重新尝试需要新的少区域tokenizer/压缩方案和独立内容/成本证据；不得仅放宽旧资格门或重复类别来制造长前缀。

十四问答案见result_answers.md；控制延迟与自然样本局限见performance_report.md；所有原始失败区域及支持不足事件在bridge_analysis.json和bridge_capture/frames.jsonl中保留。实现差异、数据/权重/hash、提交范围见delivery.json、implementation.diff、full_git.diff和artifact_sha256.json。
