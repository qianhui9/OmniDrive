# 复现与交付说明

工作目录 `/data/qh/phdCode/work4/OmniDrive`，基线 `e9e232c501da481ad5ccc8b9d2c376a2b302484c`；交付实现提交见 delivery.json。所有命令从仓库根目录执行。环境 `OmniDrive`，复用现有环境，不安装或升级依赖。

## 环境核验（独立文档验证仅执行本节）

解释器 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，Python 3.9.25，torch 1.13.1+cu117，transformers 4.31.0，mmcv-full 1.6.2，mmdet 2.28.2，mmdet3d 1.0.0rc6，numpy 1.23.4。环境声明沿用 `outputs/m3_wrcps_optimization_run01/environment_spec.json`，未重建。内核见 environment.json；完整训练与服务也已执行。

CPU 4/5 故障，所有入口必须保留 taskset，子进程继承。先检查显存；若已用显存 ≥500 MiB，等待空闲再运行 witness，不停止其他任务。sandbox 若不可见 CUDA，核验命令需要宿主 GPU 访问；这不是重装环境的理由。

```bash
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

GPU 空闲后逐字执行以下已知值 witness；预期打印 `WITNESS 510720.0 NVIDIA GeForce RTX 3090`，且 CPU affinity 不含 4/5。

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -c 'import os,torch; torch.manual_seed(0); assert not ({4,5}&set(os.sched_getaffinity(0))); x=torch.arange(64,dtype=torch.float32,device="cuda").reshape(8,8); value=float((x@x.T).sum()); assert value==510720.0; print("WITNESS",value,torch.cuda.get_device_name(0)); print("AFFINITY",sorted(os.sched_getaffinity(0)))'
```

以上是独立文档验证的全部范围；不要由验证代理启动训练或重跑整组实验。

## 固定数据与真实实验重放

现有 nuScenes 六相机图像/标注及 conv/train、conv/val，原始模型目录和 `ckpts/iter_10548.pth` 必须在原路径，具体配置见各捕获目录 effective_config.json。基础权重 SHA256 为 `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`；不把数据和基础模型放进 Git，也不自动下载。原始权重与训练 Adapter/资格 SHA 见 model_artifacts.json。

随机种子 20261028 用于资格/训练；服务按记录的 scene/frame/repeat 派生相同种子（逐帧 summary 包含 seed）。问题顺序与生成 token 原样保留。24 个资格 Log、8 个 Adapter Log、6 个验收 Log 互不交叉；历史暴露审计见 historical_data_audit.json，数据局限见 semantic_qualification_report.md。不能将回放称为新的独立验收。

所有实际 GPU/官方评估命令及返回码记录在 commands.json。冻结输入选择见 qualification_sequence.json、validation_extension_sequence.json、adaptation_sequence.json、quality_sequence.json；不要在历史输出已有的仓库重新运行 `prepare` 后声称选择相同场景。

先预览，默认不会创建输出目录或执行模型：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. /home/qh/miniconda3/envs/OmniDrive/bin/python outputs/m3_semantic_core_wrcps_run01/replay_frozen.py --root outputs/m3_semantic_core_wrcps_replay01
```

明确启动同一失败研究的完整重放（仅输出目录必须是不存在的新目录）：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. /home/qh/miniconda3/envs/OmniDrive/bin/python outputs/m3_semantic_core_wrcps_run01/replay_frozen.py --root outputs/m3_semantic_core_wrcps_replay01 --execute --diagnostic
```

脚本沿用冻结分位数规则与原始选择，执行 96 帧初始捕获、96 帧扩充捕获、资格回放、64 帧真实 Adapter 输入捕获、重建训练 manifest、3 epoch 训练、A/B 各 48 帧×3 问、官方检测/地图评估和质量分析；可选重放失败场景的 24 个同载体 Teacher 诊断。使用现有 GPU lock 和占用等待，不清理其他进程。新资格若失败即停止训练。没有调度 ExactKV 或性能准入。该脚本是复现便利工具，未重新跑完整模型组；dry-run 和编译已检查。

新目录路径进入资格 provenance，重新捕获浮点/序列化内容也可能改变文件 SHA；应重新绑定新 Adapter/资格，禁止把旧 SHA 硬写到新产物。本轮文件的原始完整性对照用 model_artifacts.json、final_source_hashes.json、raw_artifacts_sha256.json、artifact_sha256.json。

## 各阶段入口

新研究（非本轮固定数据复现）可使用 `-m tools.worldstream_m123.wrcps_semantic_experiment --root NEW --stage STAGE`：prepare → capture → qualify；初次资格支持不足时最多一次 extend-validation → qualify；只在资格通过后 prepare-adapter → capture-adapter → training-manifest → train。每个入口同样须以本文件 taskset、解释器和线程配置启动。

`-m tools.worldstream_m123.wrcps_semantic_quality --root NEW --stage zero` 执行 Native/ZeroKV 及官方评估，`--stage analyze` 重算已有 A/B 的统计。资格文件冻结后，在线代码只读取当前 M1 观测，不读取离线 GT。关键语义门失败即止步，保留正式配置。

## 回归测试

GPU 空闲后运行（606 passed；完整输出见 tests_gpu_final.log）：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123
```

无 GPU 的 sandbox 会跳过 CUDA 项，不能将其当成 GPU 通过凭据。最终测试在宿主 GPU 上执行。新增语义核心测试覆盖生命周期、资格绑定、ID 无关语义、撤销核心禁止视觉复用、固定文本保留、输入 dtype/梯度和运动关键词误报等。

## 文件范围与最终结论

完整逐帧/逐请求简表和问题原文已纳入 Git；稠密 M1 状态、真实输入张量、预测 pkl、大型官方中间预测及原图留在本服务器，SHA/大小由 raw_artifacts_sha256.json 索引。仅小型最佳 Composer adapter.pt 纳入 Git；它不是基础模型。原物理认证失败实验目录未改动。final_source_hashes.json 是最终代码，捕获目录内 source manifests 则记录当时实际执行版本；最终增加的计时字段、Anchor 撤销边界防护有回归验证，没有伪称在旧服务日志中已出现。

资格域内独立核心命中 18/126；最终 VAL 0/42。真实 Composer 已训练，四项标量质量下降在预算内，关键空间质量未通过。因此 ExactKV 与完整 E2E 加速未获得验收证据，不替换正式配置。详见 deployment_decision.json、quality_report.md、performance_report.md、result_answers.md。
