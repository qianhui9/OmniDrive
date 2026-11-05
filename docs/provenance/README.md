# 无本地 Git 历史的源码与实验溯源

用户选择将最新代码同步至 GitHub 后删除本地旧 `.git`，释放旧实验历史占用的空间。本地当前源码、模型及主实验结果继续保留；GitHub 历史不删除、不重写。

执行环境可能自动创建空 `.git` 占位目录；空目录不包含版本历史，也不作为有效 Git 仓库，版本识别仍使用独立源码清单。

`git_detachment.json` 记录原本地提交、必要历史源码的 SHA256、实验启动器迁移和当前源码清单摘要。`source_inventory.json` 用于标识独立源码版本；本地无 `.git` 时，新实验的 `source_commit` 会明确写为 `standalone-sha256:<清单摘要>`，不会冒充新的 Git 提交。

`historical_sources/` 中只有 Native 等价性和旧清理审计确实读取的五份源码快照，均为原提交的逐字节导出，并在读取时校验 SHA256。它们是审计依据，不作为推理模块导入。无需恢复整个旧仓库即可运行 Native 源码核验。

此前主实验的 `executed_source_hashes.json` 保持原字节；`frozen_main_benchmark_before_detachment.py.snapshot` 保存实验执行时的启动器。核验只允许清单固定的启动器迁移：从 Git 读取改为校验独立快照，增加独立版本标识与该迁移的校验。启动器和新增辅助模块必须与清单固定的 SHA256 一致，模型、算法、配置的任何原始校验失败仍会报错。

修改不改变模型、推理算法、参数、采样或已有实验结果。可运行：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m tools.worldstream_m123.frozen_main_benchmark verify --root outputs/m123_r0_only_reproduction_run01
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m tools.worldstream_m123.frozen_main_report --verify --root outputs/m123_r0_only_reproduction_run01
```

旧的 `git show`、`git checkout`、`git push` 等命令在本地删除 `.git` 后不再可用；它们并非推理运行命令。以后需要版本管理时，可重新克隆 GitHub 仓库到新目录，避免覆盖本地保留的模型和实验数据。
