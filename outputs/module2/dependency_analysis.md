# Dependency / equality boundary

Raw replay masks are never resized implicitly. Live M1 coordinates are the model's
preprocessed image; every pixel of a Conv2d patch must be known and independently
unchanged in the actual retained normalized input. Flow candidates cannot relocate
cached positions. Absolute/RoPE position and upstream execution signatures must match.

Any changed patch invalidates its full window; non-overlapping independent windows
can batch across cameras, with local RoPE unchanged. A global block invalidates every
token in the affected camera; initial implementation executes all cameras and all
subsequent blocks completely. A 3x3 residual bottleneck spreads one token beyond
the window and closes the reusable prefix. Patch embedding is always dense.

M1 raw coverage files report observation candidates, never neural feature equality.
See dependency_trace.json, tests, profiler linear_input_shapes and experiment_report.md
for measured closure, actual execution shapes and limitations.
