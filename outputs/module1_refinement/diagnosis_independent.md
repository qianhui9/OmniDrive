# 逐约束独立测量补充

用于诊断旧短路日志遮蔽的遮挡/外观条件；只在离线诊断中强制计算几何范围外的描述，绝不修改在线precheck判定。每个Anchor及每个Camera的原子条件保存在independent_constraint_records.json。
None表示前提缺失而无法测量，不能算成观测失败。唯一阻塞只统计其余约束全部已测量且通过。投影缺失/出视野归入observable_view；每相机projection_reason保留更细原因。诊断顺序与旧实际首失败顺序分别报告，不能混用。

## replay_strict

| 约束 | 重叠触发 | 未可测 | 唯一阻塞 | 顺序首阻塞 | 累积剩余 |
|---|---:|---:|---:|---:|---:|
| age | 0 | 0 | 0 | 0 | 468 |
| motion | 105 | 0 | 0 | 105 | 363 |
| calibration | 0 | 0 | 0 | 0 | 363 |
| observable_view | 8 | 0 | 0 | 1 | 362 |
| search_uncertainty | 98 | 8 | 0 | 0 | 362 |
| history | 0 | 8 | 0 | 0 | 362 |
| appearance | 38 | 8 | 0 | 35 | 327 |
| occlusion | 283 | 8 | 0 | 210 | 117 |
| visibility | 460 | 8 | 117 | 117 | 0 |

## replay_second_scene

| 约束 | 重叠触发 | 未可测 | 唯一阻塞 | 顺序首阻塞 | 累积剩余 |
|---|---:|---:|---:|---:|---:|
| age | 0 | 0 | 0 | 0 | 247 |
| motion | 235 | 0 | 0 | 235 | 12 |
| calibration | 0 | 0 | 0 | 0 | 12 |
| observable_view | 15 | 0 | 0 | 8 | 4 |
| search_uncertainty | 228 | 15 | 0 | 0 | 4 |
| history | 47 | 15 | 0 | 1 | 3 |
| appearance | 184 | 62 | 0 | 3 | 0 |
| occlusion | 138 | 15 | 0 | 0 | 0 |
| visibility | 232 | 15 | 0 | 0 | 0 |

## replay_interval_1s

| 约束 | 重叠触发 | 未可测 | 唯一阻塞 | 顺序首阻塞 | 累积剩余 |
|---|---:|---:|---:|---:|---:|
| age | 0 | 0 | 0 | 0 | 309 |
| motion | 73 | 0 | 0 | 73 | 236 |
| calibration | 0 | 0 | 0 | 0 | 236 |
| observable_view | 11 | 0 | 0 | 2 | 234 |
| search_uncertainty | 64 | 11 | 0 | 0 | 234 |
| history | 7 | 11 | 0 | 7 | 227 |
| appearance | 135 | 18 | 0 | 115 | 112 |
| occlusion | 183 | 11 | 0 | 71 | 41 |
| visibility | 298 | 11 | 41 | 41 | 0 |

