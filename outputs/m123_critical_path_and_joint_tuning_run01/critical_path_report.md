# Critical path evidence

Same-executor Full/Incremental initial unheavy pairs: 493.175→459.060ms visual CUDA span, 1.0743× on13 frames/one log. This is a diagnostic with lightweight observers, not an independent multi-log performance result. Known answers 31/48, same correctness in both conditions; UNKNOWN retained. Continuous output feature deviations are stored per frame.

M1 source-digest reflection and frame byte copies were low-complexity implementation opportunities; necessary state work is retained. M2 global-suffix refresh is the dominant remaining cost, so changing executor packing further has little justified headroom. Final improvements from implementation and parameters must be separated. Native host enqueue is never compared against synchronized M2 wall; native/incremental CUDA stream spans and full outer synchronized E2E are separately reported.

Instrumentation: critical_path_probe.py adds external CPU/process/thread timing wrappers, record_function ranges, cProfile and Kineto correlation summaries; critical_path.py reuses the continuous-stream full-model worker and existing Observer, captures raw predictions and frozen job manifests. Production runtime instrumentation was not added. Raw giant Chrome traces are deleted after compact summaries; temporary visual tensors remain outside Git. No per-kernel claim uses a theoretical FLOP/window estimate.
