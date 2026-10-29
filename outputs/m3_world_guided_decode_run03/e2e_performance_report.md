# Complete model E2E

432 real full model requests, 12 distinct chronological frames, three scene/log-disjoint heldout streams, three rotating A/B/C orders. Both one/three questions and natural/fixed real-image streams. All current original 513 visual tokens, six cameras, weights and sampling limits retained. First frame marked cold; cold/steady, per-scene, per-round, module CUDA/wall, FPS/QPS and peak allocation are in abc_comparison.json. Scene-bootstrap CI uses only three scene means and cannot establish dataset-wide significance. Ratios >1 favor destination; negative delta is faster.

| Workload | QA/frame | Comparison | E2E ratio | Decode ratio | E2E delta ms / scene CI95 |
|---|---:|---|---:|---:|---|
| natural | 1 | A->B | 0.997 | 0.999 | 4.50 / [-29.185090873700876, 26.7601403951024] |
| natural | 1 | B->C | 0.990 | 0.999 | 14.00 / [-0.3274921715880434, 41.319412450927] |
| natural | 1 | A->C | 0.986 | 0.998 | 18.50 / [-28.190815394433837, 57.249721488915384] |
| natural | 3 | A->B | 1.004 | 1.001 | -11.42 / [-13.960031559690833, -8.986330513531962] |
| natural | 3 | B->C | 0.993 | 1.000 | 20.26 / [15.252055561480423, 27.856523947169382] |
| natural | 3 | A->C | 0.997 | 1.000 | 8.84 / [1.2920240017895896, 16.537034457239013] |
| fixed | 1 | A->B | 1.224 | 2.300 | -232.43 / [-263.80040605242056, -216.48594760335982] |
| fixed | 1 | B->C | 0.986 | 0.999 | 14.67 / [-1.2298142925525706, 42.6239826483652] |
| fixed | 1 | A->C | 1.207 | 2.298 | -217.76 / [-221.17642340405533, -214.39150107714036] |
| fixed | 3 | A->B | 1.332 | 1.835 | -696.63 / [-859.7883958136663, -569.0408920636401] |
| fixed | 3 | B->C | 1.038 | 1.076 | -77.54 / [-177.0079591854786, -3.7798583895588913] |
| fixed | 3 | A->C | 1.383 | 1.975 | -774.17 / [-863.5682542032251, -712.8913345125815] |

Decision: **NO_GO**. M1-conditioned language-reuse objective failed: no nonexact proposals or new natural accepted tokens in the final cohort. Fixed-three-QA C is measurably faster than B, but all proposals are ordinary exact history; the gain belongs to target-aware history scheduling and changed sampled lengths, not an identified M1 semantic contribution.
