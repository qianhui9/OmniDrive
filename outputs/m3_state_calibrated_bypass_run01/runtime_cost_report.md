# Costs and unsupported-path interpretation

Task parsing is pure and bounded-LRU cached; unsupported tasks exit before target/history scans. Required AnswerMemory retrieval for the unchanged K8 target generator remains. Snapshot detector-center projection is CPU-only and distinct from historical reuse qualification. No extra neural forward. Per-question parsing, binding, temporal, validity, construction, RNG management and AnswerMemory costs plus all M1/M2/Heads/CUDA spans are in abc_comparison.json. Nested binding/temporal costs must not be summed again.

- addressable 1Q A: E2E 1115.808ms; M1 193.009, M2 433.076, heads 15.105; decision on misses 0.0ms; RNG management Nonems/Q.
- addressable 1Q B: E2E 946.923ms; M1 170.749, M2 424.132, heads 15.075; decision on misses 0.0ms; RNG management 0.07590333310266335ms/Q.
- addressable 1Q C: E2E 855.371ms; M1 194.075, M2 423.254, heads 15.082; decision on misses 0.028465370026727516ms; RNG management 0.03768339714345833ms/Q.
- addressable 3Q A: E2E 2425.990ms; M1 190.191, M2 426.610, heads 15.222; decision on misses 0.0ms; RNG management Nonems/Q.
- addressable 3Q B: E2E 1741.646ms; M1 207.193, M2 424.510, heads 15.133; decision on misses 0.0ms; RNG management 0.065945399304231ms/Q.
- addressable 3Q C: E2E 1270.562ms; M1 177.649, M2 423.976, heads 15.158; decision on misses 0.02268326675726308ms; RNG management 0.03528261246780554ms/Q.
- fixed 1Q A: E2E 1398.155ms; M1 190.506, M2 434.748, heads 15.191; decision on misses 0.0ms; RNG management Nonems/Q.
- fixed 1Q B: E2E 1039.913ms; M1 189.038, M2 424.256, heads 15.103; decision on misses 0.0ms; RNG management 0.07559896524374683ms/Q.
- fixed 1Q C: E2E 1028.608ms; M1 178.504, M2 424.471, heads 15.800; decision on misses 0.002562854206189513ms; RNG management 0.0750545586925ms/Q.
- fixed 3Q A: E2E 3019.657ms; M1 192.864, M2 426.208, heads 15.139; decision on misses 0.0ms; RNG management Nonems/Q.
- fixed 3Q B: E2E 1873.962ms; M1 193.772, M2 425.457, heads 15.045; decision on misses 0.0ms; RNG management 0.06631515053514805ms/Q.
- fixed 3Q C: E2E 1888.223ms; M1 189.558, M2 435.130, heads 15.149; decision on misses 0.0023100195297350488ms; RNG management 0.06624196410282618ms/Q.
- mixed 3Q A: E2E 2790.640ms; M1 208.914, M2 425.275, heads 15.222; decision on misses 0.0ms; RNG management Nonems/Q.
- mixed 3Q B: E2E 2597.934ms; M1 198.313, M2 425.600, heads 15.171; decision on misses 0.0ms; RNG management 0.07135344074211186ms/Q.
- mixed 3Q C: E2E 2437.220ms; M1 195.943, M2 426.132, heads 15.163; decision on misses 0.008350268739168762ms; RNG management 0.06238583268390761ms/Q.
- natural 1Q A: E2E 1453.759ms; M1 164.885, M2 438.100, heads 15.479; decision on misses 0.0ms; RNG management Nonems/Q.
- natural 1Q B: E2E 1430.146ms; M1 173.047, M2 431.237, heads 15.224; decision on misses 0.0ms; RNG management 0.08162669837474823ms/Q.
- natural 1Q C: E2E 1413.246ms; M1 164.299, M2 422.198, heads 15.451; decision on misses 0.040808391834919654ms; RNG management 0.08227991444679598ms/Q.
- natural 3Q A: E2E 2973.561ms; M1 210.785, M2 425.526, heads 15.191; decision on misses 0.0ms; RNG management Nonems/Q.
- natural 3Q B: E2E 2921.256ms; M1 180.992, M2 427.155, heads 15.135; decision on misses 0.0ms; RNG management 0.07480520677442352ms/Q.
- natural 3Q C: E2E 2922.465ms; M1 185.291, M2 424.488, heads 15.074; decision on misses 0.013372970796707604ms; RNG management 0.07470317521236008ms/Q.

The tens-of-microseconds decision cost cannot explain all wall-time deltas. Stage means and scene-paired intervals expose M1/M2/CUDA/order variability; no claim of pure parser causality is made. Native RNG management is not separately instrumented, though its outer E2E includes it.
