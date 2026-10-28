# Complete M123 E2E

P2 retained requests: 432 (single/three QA; zero means not rerun). Final P3 requests: 216 (three QA). Both use three independent scene/log streams, four distinct chronological frames per scene, three rotating A/B/C orders. All original six cameras, native Object256+Ego1+Map256, current visual prefill, original questions and original sampling contract are retained. Natural QA uses original dataset questions; fixed QA repeats source-frame actual questions on changing images. Each condition has its OWN earlier model answers, never A's current answer. Each scene resets identical M1/M2/head memory; the first frame is separately marked cold.

All variants time the same outer model invocation, synchronize before/after and include M1/M2/heads, current assembly, language runtime, C snapshot/retrieval/diff and answer serialization. Data loading/collation and offline analysis are outside equally. No pre-expanded-carrier generation wall is substituted for complete M123 wall. C's post-commit snapshot cost is included in E2E and reported separately from M1 commit wall.

Ratios >1 favor the destination. Delta is destination minus source, so negative is faster. CI resamples paired scene means (three scenes only); this is limited evidence, not full nuScenes deployment coverage.

| Workload | Q/frame | Comparison | E2E ratio | M3 ratio | Decode wall ratio | E2E change ms [scene CI95] |
|---|---:|---|---:|---:|---:|---|
| natural | 3 | A->B | 0.989 | 0.995 | 1.000 | 35.03 [6.511136268575986, 60.076479062748454] |
| natural | 3 | B->C | 1.002 | 0.994 | 0.992 | -4.86 [-33.800910653856896, 17.167362268082798] |
| natural | 3 | A->C | 0.991 | 0.989 | 0.992 | 30.17 [8.55137970453749, 55.67981108712653] |
| fixed | 3 | A->B | 1.342 | 1.531 | 1.929 | -710.22 [-765.7926400036862, -651.4768456496919] |
| fixed | 3 | B->C | 0.973 | 0.944 | 0.907 | 58.52 [3.278001871270438, 91.18957604126383] |
| fixed | 3 | A->C | 1.305 | 1.445 | 1.750 | -651.70 [-710.1164509076625, -560.2872696084281] |

P2 initial-policy controls, including single-QA (not final-policy single-QA qualification):

| P2 workload | Q/frame | Comparison | E2E ratio |
|---|---:|---|---:|
| natural | 1 | A->B | 0.997 |
| natural | 1 | B->C | 0.999 |
| natural | 1 | A->C | 0.995 |
| natural | 3 | A->B | 0.994 |
| natural | 3 | B->C | 0.995 |
| natural | 3 | A->C | 0.989 |
| fixed | 1 | A->B | 1.167 |
| fixed | 1 | B->C | 0.860 |
| fixed | 1 | A->C | 1.004 |
| fixed | 3 | A->B | 1.341 |
| fixed | 3 | B->C | 0.747 |
| fixed | 3 | A->C | 1.002 |

Decision **NO_GO**: M1-guided C is reproducibly slower than history-only B in at least one primary workload; semantic candidates did not establish independent E2E benefit. Cold/steady, per-scene/per-round values, module spans, output tokens, FPS/QPS and peak GPU allocation are in abc_comparison.json. Sampling output lengths vary; raw E2E and per-token/forward evidence must be read together. No answer truncation/forced token length is used.
