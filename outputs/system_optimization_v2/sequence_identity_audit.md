# 序列身份审计

根因：Dataset 按时间全局排序，旧报告将 Dataset 下标用于未排序的原始 Pickle。
按 token 连接并核对 scene/timestamp 后，实际执行连续；修正覆盖字段，不否定或改写原时延。

| Run | Scene | 原 duration s | 修正 duration s | 原 path m | 修正 path m |
|---|---|---:|---:|---:|---:|
| budget_0 | 9f1f69646d644e35be4fe0122a8b91ef | -1316169.649495 | 0.399799 | 1017.920 | 1.763 |
| budget_0 | a04daf2d0f194b2ab2ff2a47dfebc1d7 | 0.499931 | 0.499852 | 0.015 | 2.290 |
| budget_128 | 9f1f69646d644e35be4fe0122a8b91ef | -1316169.649495 | 0.399799 | 1017.920 | 1.763 |
| budget_128 | a04daf2d0f194b2ab2ff2a47dfebc1d7 | 0.499931 | 0.499852 | 0.015 | 2.290 |
| language_candidates | 9f1f69646d644e35be4fe0122a8b91ef | -1316168.650279 | 1.499610 | 1025.004 | 6.795 |
| language_candidates | a04daf2d0f194b2ab2ff2a47dfebc1d7 | 1.500899 | 1.500161 | 0.263 | 7.215 |
| lifecycle_validation | a178a1b5415f45c08d179bd2cacdf284 | 0.949993 | 0.950513 | 6.783 | 0.317 |
| lifecycle_validation | a2b005c4dd654af48194ada18662c8ca | 1.099868 | 1.000275 | 8.505 | 7.095 |
| long_stream_retry | a178a1b5415f45c08d179bd2cacdf284 | 15.249527 | 15.600126 | 50.414 | 29.712 |
| long_stream_retry | a2b005c4dd654af48194ada18662c8ca | 15.499515 | 15.349626 | 135.310 | 94.560 |
| multi | 9f1f69646d644e35be4fe0122a8b91ef | -1316168.650279 | 1.499610 | 1025.004 | 6.795 |
| multi | a04daf2d0f194b2ab2ff2a47dfebc1d7 | 1.500899 | 1.500161 | 0.263 | 7.215 |
| single | 9f1f69646d644e35be4fe0122a8b91ef | -1316168.650279 | 1.499610 | 1025.004 | 6.795 |
| single | a04daf2d0f194b2ab2ff2a47dfebc1d7 | 1.500899 | 1.500161 | 0.263 | 7.215 |
| smoke_single_fixed | 9f1f69646d644e35be4fe0122a8b91ef | -1316169.649495 | 0.399799 | 1017.920 | 1.763 |
| stride_refresh | 9f1f69646d644e35be4fe0122a8b91ef | -1316167.250727 | 2.999240 | 1033.074 | 14.804 |
| stride_refresh | a04daf2d0f194b2ab2ff2a47dfebc1d7 | 3.001199 | 2.999309 | 3.449 | 15.582 |
