# Measurement boundaries

The native profiling pass captures real 513-token carriers produced by the existing M1/M2/object/map pipeline on 24 distinct sequential nuScenes samples. It measures the first three original conversation questions per frame without changing their wording or order. Native generation uses the existing pretrained model and its configured sampling defaults, not fixed output lengths.

Forward CUDA events measure each actual target invocation; decode wall is the stream span from completed prefill to completed generation (including CPU gaps). Host enqueue time is recorded separately. Head and M1/M2 stage wall values are instrumented host durations; they can include CUDA synchronizations performed by the original code. They are not presented as isolated kernel times.

Cold first frame is retained and tagged warmup. Subsequent paired decoder trials use the very same captured current-frame carriers for A and B, freshly prefill each question, and never inherit decode K/V from a preceding frame. Captured-carrier timing does not establish full M123 end-to-end speedup. Fixed-question trials repeat original first-frame question IDs on **different** images from the same scene.

Natural strict-question availability uses earlier-frame questions only. Within-frame questions do not create cross-frame history opportunities. Exact and normalized-template matching are separate diagnostics. Template matching is not a license to change the question or assume answer equivalence. History contains only model-generated token IDs, never annotation answers.

A first capture attempt failed after the first model request because the model changes the timestamp input container in place. The capture now uses the dataset timestamp metadata; the failed attempt directory and log are retained and excluded from all results.
