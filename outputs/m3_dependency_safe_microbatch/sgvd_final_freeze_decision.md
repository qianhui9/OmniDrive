# SGVD final freeze

native_fp16_execution_shape_rounding

Original FP16 arithmetic and sampling settings are unchanged. SGVD remains disabled. A mathematical acceptance law does not certify logits produced by a different GEMM/Attention shape. No larger draft search or SGVD generation runtime is introduced.
