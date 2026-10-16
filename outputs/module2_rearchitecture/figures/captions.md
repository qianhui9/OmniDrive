All values are read from executed experiment JSON; no theoretical FLOPs converted to latency.

feasibility: idealized current-oracle bypass, 10 measurements after 3 warmups, RTX 3090; not historical reuse evidence. M1 line is this round's six-scene median for the preserved prior M1+M2 code.

numerical_propagation: independent seeds 2801/3509, same patch change, original full vs dependency-sparse execution; no tolerance change. All-window packed comparison separately has zero difference in these inputs.

real_ablation: six independent real driving scenes, 24 unique frames, two measured replays after one warmup; bars P50 with upper P95. Reset/output-only alter WorldState history contract and do not represent continuity-preserving M1 optimizations. Complete heads/temporal state/greedy LLM execute.

state_drift: 16 chronological pose/time steps in each of two scenes. Curves show worst same-class matched centre error over scenes; mixed inputs are synthetic, repeated inputs preserve real varying pose/time. Neither is real camera reuse coverage or broad safety certification.

camera_boundary: left actual runtime cost includes exact validation/cache/assembly, but partial counts remain experimental unless quality qualified. Right final formal Auto and Auto with continuous M1, balanced mode order and two measured replays after one warmup; not real driving benefit.
