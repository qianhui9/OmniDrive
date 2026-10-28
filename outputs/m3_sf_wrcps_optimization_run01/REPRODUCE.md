# SF-WRCPS 复现说明

仓库根目录 `/data/qh/phdCode/work4/OmniDrive`，参考提交 `0932a377b199e6249b87e15881c6e3c960f3cf3b`，本轮分支 `m3-sf-wrcps-20261028-run01`。最终实现提交见 delivery.json。已有 OmniDrive 环境原样复用，未安装或升级包。

## 环境核验（独立文档验证只执行本节）

所有命令从仓库根目录执行。CPU 4/5 故障，必须保留 taskset；子进程继承亲和性。解释器 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。环境声明 `outputs/m3_wrcps_optimization_run01/environment_spec.json`，SHA256 `eda83d61075d3610f664d5edffd4c5bf070c6dfed0822bf086252977b5c3e2e5`，沿用 `.aris/compute/local.md`。Python 3.9.25、torch 1.13.1+cu117、transformers 4.31.0；实际版本与 witness 见 environment.json。

先逐字执行：

```bash
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

仅当显存已用量 <500 MiB 时执行以下已知值 witness；否则等待，不停止任何其他进程。sandbox 看不到 CUDA 时需宿主 GPU 访问，无须重装环境。

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -c 'import os,torch; torch.manual_seed(0); assert not ({4,5}&set(os.sched_getaffinity(0))); x=torch.arange(64,dtype=torch.float32,device="cuda").reshape(8,8); value=float((x@x.T).sum()); assert value==510720.0; print("WITNESS",value,torch.cuda.get_device_name(0)); print("AFFINITY",sorted(os.sched_getaffinity(0)))'
```

预期 `WITNESS 510720.0 NVIDIA GeForce RTX 3090`，亲和性不含 4/5。以上是独立文档验证全部范围，不启动模型、训练或完整回归。

## 数据、权重与冻结选择

使用服务器现有 nuScenes v1.0-trainval、六相机图像、conv/train 与 conv/val、原模型目录和 `ckpts/iter_10548.pth`。不自动下载。基础 checkpoint SHA256 `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`，已在真实无核心诊断时重新计算。资格、Adapter、来源文件摘要见 model_artifacts.json 与 final_source_hashes.json。禁止将 GT 加入在线稳定状态。

冻结输入见 qualification_train_sequence.json（8 Log /64帧）、qualification_val_sequence.json（6 Log /48帧）、adaptation_sequence.json（10 Log /80帧，6训练/4验证）、target_sequence.json（6 Log /48帧）。每帧均保留原始前三个问题及顺序，生成配置不缩短。资格为7校准+7独立验证Log，训练、开发验证与最终保留Log不交叉。历史暴露局限见 historical_data_audit.json 和 experiment_plan.json；最终6个VAL Log中5个曾被更早M3研究使用，不能称全部历史完全未见。

不要在已有输出的仓库重新执行 prepare 后声称复现同一选择。以下脚本直接复制冻结的选择，默认只预览命令，不创建输出目录：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. /home/qh/miniconda3/envs/OmniDrive/bin/python outputs/m3_sf_wrcps_optimization_run01/replay_frozen.py --root outputs/m3_sf_wrcps_replay01
```

明确重放开发阶段（新输出目录必须不存在）：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. /home/qh/miniconda3/envs/OmniDrive/bin/python outputs/m3_sf_wrcps_optimization_run01/replay_frozen.py --root outputs/m3_sf_wrcps_replay01 --execute --through core-quality
```

执行112帧资格采集、CAL-only规则选择、一次独立资格验证；若通过且独立VAL存在正确核心命中，再采集80帧开发数据、冻结基础权重训练K64 Composer、评估全部开发验证问题的同载体Native/SF-ZeroKV。GPU作业使用已有互斥锁并等候显存，不停止其他任务。独立训练/验证都没有真实核心时训练拒绝执行。空间回归不得通过增加epoch或修改最终保留集补救。

新目录路径、重新采集结果和浮点序列化可能改变资格/Adapter SHA；必须由新训练过程重新绑定，禁止硬写本轮旧摘要。replay脚本不自动进行最终验收或部署；真实有效核心质量须先审阅，再允许打开保留数据。

## 表示诊断与阶段入口

所有以下命令也须使用同一 taskset、解释器和线程配置。`-m tools.worldstream_m123.sf_wrcps_experiment --root NEW --stage prepare|capture|qualify|adaptation` 用于新研究；冻结car门槛不变，新类型仅三组CAL分位数规则。已有资格文件禁止覆盖。

`-m tools.worldstream_m123.sf_wrcps_diagnostics --root NEW --stage structure` 使用旧真实开发载体，比较原生槽位保留；新压缩器未训练，不能作为语言质量证据。

`--stage no-core` 重放旧故障场景8帧24问，使用原始采样状态，比较真实扩展embedding、position/mask、prefill logits和完整生成tokens。依赖 `outputs/m3_semantic_core_wrcps_run01/failure_diagnostic/m3_inputs/*.pt`，摘要索引见原实验文件；旧场景只作为回归，不能当作新独立验收。

`--stage core-quality` 在真实已训练Adapter上评估整个开发验证序列，逐问题使用同一当前载体、问题和RNG，禁止历史KV。记录参考/Teacher/SF答案，自动解析只作审阅辅助。`core_valid_quality/critical_review.json` 必须绑定当前Adapter并明确开发核心质量通过后，才可运行 `-m tools.worldstream_m123.sf_wrcps_quality --root NEW` 打开最终保留集。四项标量预算与关键语义均通过后，才有ExactKV/性能准入资格。

本轮实际命令、返回码、日志见 commands.json、各driver日志和各capture/invocation.json。最终研究结果和停止阶段见 deployment_decision.json；未执行阶段的指标必须保持null，不能从旧实验搬入。

## 回归

GPU空闲后运行以下命令。sandbox CUDA跳过不能作为CUDA通过凭据；本轮宿主执行证据为 tests_gpu_final.log 和 final_test_result.json。

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123
```

原始权重、M1/M2主算法、PrefillExecutor与正式配置保持原样。小型最佳Adapter若实际训练完成可纳入Git；基础权重、真实载体张量、稠密状态、预测中间文件仅保留服务器并由 raw_artifacts_sha256.json 索引。正式替换仅在全部验收通过后推荐。
