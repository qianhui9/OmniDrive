# Implementation and instrumentation changes

| File | Type | Purpose | Lines |
|---|---|---|---|
| worldstream/m3/config.py | modified | Runtime opt-in execution and conservative Prefix guard | 1–59 |
| worldstream/m3/runtime.py | modified | Runtime opt-in execution and conservative Prefix guard | 1–337 |
| worldstream/m3/native_decode.py | modified | Runtime opt-in execution and conservative Prefix guard | 1–183 |
| tools/worldstream_m3/decode_optimization/service.py | created | Standalone real-data measurement/reproduction or contract regression | 1–125 |
| tools/worldstream_m3/decode_optimization/attribution.py | created | Standalone real-data measurement/reproduction or contract regression | 1–58 |
| tools/worldstream_m3/decode_optimization/sampling_prefix.py | created | Standalone real-data measurement/reproduction or contract regression | 1–82 |
| tools/worldstream_m3/decode_optimization/profile.py | created | Standalone real-data measurement/reproduction or contract regression | 1–123 |
| tools/worldstream_m3/decode_optimization/batch.py | created | Standalone real-data measurement/reproduction or contract regression | 1–42 |
| tools/worldstream_m3/decode_optimization/micro.py | created | Standalone real-data measurement/reproduction or contract regression | 1–111 |
| tools/worldstream_m3/decode_optimization/prefix.py | created | Standalone real-data measurement/reproduction or contract regression | 1–129 |
| tools/worldstream_m3/decode_optimization/confirmation.py | created | Standalone real-data measurement/reproduction or contract regression | 1–68 |
| tools/worldstream_m3/decode_optimization/__init__.py | created | Standalone real-data measurement/reproduction or contract regression | 1–1 |
| tools/worldstream_m3/decode_optimization/replay.py | created | Standalone real-data measurement/reproduction or contract regression | 1–18 |
| tools/worldstream_m3/decode_optimization/finalize.py | created | Standalone real-data measurement/reproduction or contract regression | 1–173 |
| tests/worldstream_m3/test_decode_graphs.py | created | Standalone real-data measurement/reproduction or contract regression | 1–126 |

Instrumentation is confined to standalone runners. M1/M2/original model/TaskEvidence source hashes are checked in original_model_parity.json. Temporary module installation is restored in finally.

decode_cuda_graph=False preserves the original route. identical_prefill_only=False preserves the legacy research route; True rejects partial-prefix reuse without relaxing Auto certificates. Both are independent of Direct Query gates.
