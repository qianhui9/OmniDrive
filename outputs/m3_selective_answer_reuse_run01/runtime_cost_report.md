# Runtime costs

For the formal M1-addressable three-question C workload, mean exclusive fact parsing/check/construction costs are 0.0419/0.0362/0.0820 ms per question (sum 0.1601 ms). Current snapshot mean is 0.8879 ms per frame. Model binding, prompt transfer, memory management and output serialization remain included in measured outer E2E. These isolated CPU phases are not substituted for full E2E.

Question-local wall is the M3 request span. One-question full M123 E2E is the outer synchronized frame wall for question_count=1; three-question full E2E uses question_count=3. Prefill host enqueue and CUDA spans are distinct. Fact checking uses CPU scalar/tuple state; constructing the final output tensor enqueues a small device copy, whose completion is included by the outer CUDA synchronization.
