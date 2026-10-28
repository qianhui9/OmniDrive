# SF-WRCPS 训练与收益可行性复现

仓库 `/data/qh/phdCode/work4/OmniDrive`；基线 `ec4c667736a72afb2cafb1635540d9db613f333c`。本轮分支 `m3-sf-wrcps-feasibility-20261028-run01`，实现提交见 delivery.json。沿用现有K64 Composer、M1/M2和PrefillExecutor；正式配置不改。旧实验未覆盖。

## 环境核验（独立文档验证仅执行本节）

现有OmniDrive环境，解释器 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，Python3.9.25、torch1.13.1+cu117、transformers4.31.0、RTX3090 24GB；未安装依赖或重建环境。环境声明 `outputs/m3_wrcps_optimization_run01/environment_spec.json`，SHA256 `eda83d61075d3610f664d5edffd4c5bf070c6dfed0822bf086252977b5c3e2e5`，账本 `.aris/compute/local.md`。所有命令从仓库根目录执行，CPU4/5故障，必须保留taskset。

先逐字执行：

```bash
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

仅当显存已用<500MiB时执行以下命令；否则等待，不停止其他任务。sandbox不可见CUDA时使用宿主GPU访问，不重装环境。

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -c 'import os,torch; torch.manual_seed(0); assert not ({4,5}&set(os.sched_getaffinity(0))); x=torch.arange(64,dtype=torch.float32,device="cuda").reshape(8,8); value=float((x@x.T).sum()); assert value==510720.0; print("WITNESS",value,torch.cuda.get_device_name(0)); print("AFFINITY",sorted(os.sched_getaffinity(0)))'
```

预期 `WITNESS 510720.0 NVIDIA GeForce RTX 3090`，affinity不含4/5。独立文档验证不得启动模型、训练或测试，只执行本节两条命令。

## 数据和执行顺序

使用服务器现有nuScenes官方TRAIN元数据、六相机图像、conv/train原始问答、原始模型目录与 `ckpts/iter_10548.pth`。不下载数据和权重。基础权重SHA256 `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`；本轮micro重新校验。沿用 `outputs/m3_sf_wrcps_optimization_run01/semantic_qualification.json`，SHA256 `290c5d88da0795122c9e1cd79a3ae808ba51507e56bb8b9683bc1126909aee65`，不拟合或放宽规则。

全TRAIN池元数据审计：700场景、28130帧、50Log；严格排除当前资格校准/验证、开发验证及最终保留Log。允许可追溯旧TRAIN研究数据用于训练。冻结576帧名单在 audit_sequence.json：原6训练Log+8个按既有真实核心证据选择的合法训练Log，各前32帧；开发验证维持原4Log，各前32帧。完整32帧因果执行后比较8/16/32前缀，不在每个窗口重置来伪造连续性。历史使用与保护划分见 official_train_pool_audit.json、training_historical_exposure.json。

所有GPU入口都使用已有gpu_job_lock与显存等待。状态审计保持原生感知、M1关联和M2计算；只在工具调用中清空语言问题，保留原始前三个问题编码和载体，避免重复Decode。它不是正式VLM效率/质量实验。每帧状态只来自当前和之前真实图像；GT评分在模型commit之后单独执行，从不进入M1/M3。

新研究入口（NEW目录必须不存在）：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.sf_feasibility --root NEW --stage prepare
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.sf_feasibility --root NEW --stage audit
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.sf_feasibility_analysis --root NEW --stage data
```

精确重放本轮选择时，NEW目录只复制 experiment_plan.json、audit_sequence.json、official_train_pool_audit.json、profit_gate_definition.json，跳过prepare，随后执行audit/data。不要修改已存在的原始证据目录。prepare依赖历史索引，未来输出改变后重新选择不能称为本轮固定样本复现。

## CUDA上限及语义诊断

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.sf_feasibility_micro --root NEW
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.sf_feasibility_analysis --root NEW --stage estimate
```

micro依赖旧SF开发验证捕获 `outputs/m3_sf_wrcps_optimization_run01/adaptation_capture/m3_inputs/*.pt`。只读已暴露的真实有效核心相邻帧；原生LLM、既有PrefillExecutor，固定当前输入比较Native、full、text、text+64、同帧visual五种硬件控制。2轮预热、9轮有效测量、三个原始问题、交替顺序。每次只生成一个token、零Decode Forward，是明确的prefill硬件测量，不能冒充自然生成服务加速。Exact输入的embedding/position/mask与缓存prefix严格相等，由原有cache/executor验证。实际现有Scheduler决策另列，控制长度不会写入正式policy。

Composer随机未训练参数只用于实际K64结构成本、Stable编码恒等审计和有限反事实影响诊断，禁止当作正式质量或加速证据。语义反事实仅改Stable64，Dynamic、布局、问题及RNG固定；另列Native回答与参考。自然诊断生成沿用温度0.1/top_p0.75/max_new_tokens320，无历史KV。

期望收益使用旧独立资格VAL的真实生命周期和单entry/layout状态回放，区分每帧1问/3问、跨帧Stable64、固定文本和同帧visual；实际Scheduler缺乏新布局质量/成本绑定时准入为0。乐观匹配机会不等于真实Executor服务复用。净收益用实测Native与cache-owning Executor的Wall差计算，不能将共同的原生生成开销重复算作新成本。profit_gate_definition.json在micro执行前冻结筛选规则；准入要求单问总净值及世界前缀边际净值的95%计时下界都为正；bootstrap只描述micro测量不确定性，不是部署证明或总体无收益证明。

## 条件性训练、质量与停止

只有 feasibility_decision.json 的data_go和speed_go均通过，才调用原有 `wrcps_train.py --manifest NEW/training_dataset_manifest.json --output NEW/adapter_training --spatial --stable-tokens 64 --rank 128 --epochs 3 --seed 20261028`。训练只更新Composer，基础权重冻结；真实无核心无虚假更新；按Core-Valid开发验证loss选checkpoint。训练事件富集不重复帧，不改变在线快照；开发验证保留全部32帧。富集命中率不作为自然部署覆盖率。

训练后继续使用已有 `sf_wrcps_diagnostics --stage core-quality` 和 `sf_wrcps_quality`。开发关键语义审阅、整体四指标及关键语义通过后才有正式ExactKV/E2E准入。未执行阶段必须报告NOT_RUN/null，不能填旧实验值。最终保留数据始终沿用旧target_sequence.json，不用于训练或选择。

## 回归与交付

GPU空闲后：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123
```

实际CUDA日志见 tests_gpu_final.log。稠密state_audit/frames.jsonl和真实载体.pt留在服务器，由raw_artifacts_sha256.json索引；精简事件、数据manifest、micro逐条计时、全部报告、日志及Git差异纳入交付。模型与资格摘要见model_artifacts.json，15项回答见result_answers.md。
