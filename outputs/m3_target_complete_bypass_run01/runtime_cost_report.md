# Measured execution costs

C-target alone reads already decoded competition scalars inside M1 output adapter and projects centers in snapshot. It does not alter M1 filtering/association, M2 reuse or any model forward. Basic/B do not pay the additional readout cost. Unsupported requests exit before target checks; normal K8 memory and RNG work remain.

- addressable1Q A: E2E=1177.6992ms/frame, snapshot=0.0000, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- addressable1Q B: E2E=1007.7080ms/frame, snapshot=1.0549, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- addressable1Q C: E2E=895.9868ms/frame, snapshot=1.0636, parsing=0.003472ms/Q, extraction=0.018860, target check=0.000000, miss decision=0.03380967038018363ms.
- addressable3Q A: E2E=2529.5113ms/frame, snapshot=0.0000, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- addressable3Q B: E2E=1758.1771ms/frame, snapshot=1.1289, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- addressable3Q C: E2E=1307.7873ms/frame, snapshot=1.0556, parsing=0.001885ms/Q, extraction=0.016613, target check=0.000000, miss decision=0.03208116419157203ms.
- fixed1Q A: E2E=1364.1226ms/frame, snapshot=0.0000, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- fixed1Q B: E2E=1110.2591ms/frame, snapshot=1.0552, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- fixed1Q C: E2E=1136.0962ms/frame, snapshot=1.1449, parsing=0.001976ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0029845250537618995ms.
- fixed3Q A: E2E=2971.3580ms/frame, snapshot=0.0000, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- fixed3Q B: E2E=1987.1292ms/frame, snapshot=1.0550, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- fixed3Q C: E2E=1985.8959ms/frame, snapshot=1.0803, parsing=0.001633ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0025915796868503094ms.
- mixed3Q A: E2E=2723.9205ms/frame, snapshot=0.0000, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- mixed3Q B: E2E=2479.9714ms/frame, snapshot=1.0829, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- mixed3Q C: E2E=2221.4439ms/frame, snapshot=1.0684, parsing=0.001527ms/Q, extraction=0.006869, target check=0.000000, miss decision=0.007896128898629775ms.
- natural1Q A: E2E=1363.6375ms/frame, snapshot=0.0000, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- natural1Q B: E2E=1284.1925ms/frame, snapshot=1.0152, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- natural1Q C: E2E=1288.2892ms/frame, snapshot=1.1078, parsing=0.067238ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.06821243732701987ms.
- natural3Q A: E2E=2774.5485ms/frame, snapshot=0.0000, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- natural3Q B: E2E=2722.9157ms/frame, snapshot=1.1102, parsing=0.000000ms/Q, extraction=0.000000, target check=0.000000, miss decision=0.0ms.
- natural3Q C: E2E=2727.8163ms/frame, snapshot=1.0448, parsing=0.022586ms/Q, extraction=0.000366, target check=0.000000, miss decision=0.023110480488915192ms.

Extraction and completeness are nested inside validity; do not sum them again. M1/M2/Heads and enqueue/CUDA/outer wall spans differ; complete E2E variation cannot be attributed solely to microsecond parsing. FPS/QPS, memory, forwards and phase timings are in abc_comparison.json.
