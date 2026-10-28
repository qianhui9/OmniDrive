# 复现WRCPS run01

目录：`/data/qh/phdCode/work4/OmniDrive`。分支 `m3-wrcps-20261028-run01`。起点 `268e0d2b`；最终代码和交付提交见 `delivery.json` / Git HEAD。保留原有模型、M1/M2和正式配置。所有执行均继承CPU亲和性 `0-3,6-31`，禁止CPU4/5。无需重建环境、下载权重或修改数据。

## 计算环境见证（已执行）

先只读检查GPU，其他任务占用时等待，不能结束其进程：

```sh
taskset -c 0-3,6-31 nvidia-smi
```

在已有OmniDrive环境运行：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -c 'import torch; torch.manual_seed(20261028); x=torch.arange(64,device="cuda",dtype=torch.float32).reshape(8,8); value=float((x@x.T).sum()); assert value==510720.; print("WRCPS_ENV_OK", value, torch.cuda.get_device_name(0))'
```

预期：`WRCPS_ENV_OK 510720.0 NVIDIA GeForce RTX 3090`。环境规格和实际包版本见 `environment_spec.json` / `environment.json`。

## 当前审计的精确重放

原始NuScenes六相机数据、`ckpts/iter_10548.pth`、项目已有tokenizer与依赖必须存在。先复制冻结输入到新目录；重放相同8场景不能当作新增8个独立场景：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python - <<'PY'
from pathlib import Path
import shutil
source=Path('outputs/m3_wrcps_optimization_run01')
target=Path('outputs/m3_wrcps_reproduce_run01')
target.mkdir(exist_ok=False)
for name in ('anchor_sequence.json','experiment_plan.json','historical_data_audit.json'):
    shutil.copy2(source/name,target/name)
PY
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.wrcps_experiment --root outputs/m3_wrcps_reproduce_run01 --stage audit
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.wrcps_experiment --root outputs/m3_wrcps_reproduce_run01 --stage analyze
```

Audit使用共享GPU锁与显存空闲检查，单一正式M123路径执行48帧，每帧原始顺序前三个自然问题。Analyze检查完整成功状态、帧序和快照Hash后生成JSON统计及逐请求记录。实际初次成功审计直接调用同一个worker；精确命令和退出状态见 `commands.json`。首个失败的无问题调用保留在 `anchor_audit.log`，不进入成功统计。

如果需要新场景，而非精确重放，在未存在的输出根目录运行 `--stage prepare`，然后audit/analyze。Prepare依据历史M3语言记录排除已见场景，选2个开发和6个独立场景，每场景6个相邻prev/next可核实的帧。

## 回归测试（已执行）

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q
```

GPU可访问时594项通过，见 `tests_gpu.log`。合成小Llama测试使用3文本+128稳定视觉token验证131-token Exact Prefix、真实386-token后缀forward和ZeroKV生成一致性；文本前缀长度3是测试夹具，不能当作真实系统35-token结果。测试中的稳定状态/Adapter为合成夹具，未计入真实结果、训练数据或性能证据。

## 未执行阶段

真实审计结果：48帧均无可用认证稳定事实，0/40跨帧Anchor命中。没有Adapter训练、ZeroKV四指标评估、真实WRCPS ExactKV或A/B/C完整请求性能测量。`training_report.md`说明带前置条件的训练接口；`validation_plan.json`仅保存未来验收合同，不代表已执行。正式配置没有指向实验Adapter或新policy。

`--wrcps-adapter`只为已有训练权重启用新布局；必须配 `--m3-strategy exact_prefix`。新的policy必须绑定该Adapter Hash和新执行/布局签名。没有合法新证据时调度器不会复用。旧policy不能靠复制或改标签变成新布局的质量证书。

## 产物与来源

`stable_anchor_analysis.json`、`m1_qualification_audit.json`、`stable_anchor_timeline.jsonl`是世界状态证据；`per_request_metrics.jsonl`是144条原生布局审计请求，明确标注wrcps_executed=false。设计、训练、质量、性能与不部署结论见对应报告。

原始M1密集日志、预测、基础权重和Adapter文件均留服务器，不提交相机图像或私有原始数据；Hash索引见 `raw_artifacts_sha256.json`。代码与报告Diff见 `implementation.diff`。

成功审计运行与最终代码整理有时间重叠，运行开始时的源码Hash保存在 `anchor_audit_language/m1_m2_m3_source_hashes.json` 和benchmark Hash文件中。该审计执行原有native布局，WRCPS从未被启用；最终额外的稳定快照诊断字段不进入既有48帧数据。完整最终源码由CUDA回归验证。不能把该原生路径审计当作最终WRCPS性能测量。
