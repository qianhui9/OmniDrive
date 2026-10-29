# Question RNG isolation

Old outer worker seeds each frame; native torch.multinomial and deterministic-draft rejection sampler consume the shared default generator. Per-request fork_rng restores outer state and re-seeds by stable scene/frame/index/repeat/experiment SHA256.

The historical right-rear/right-front difference is caused by shifted sampling RNG: identical current embeddings and draft candidates, with both crossed RNG interventions reproducing the corresponding exact historical answer. Question isolation removes the between-condition difference, not the target model error; both isolated outputs are left-front while current GT is right-rear.

Same-input/draft RNG intervention changes legacy output: True. Isolated-state crossover equality: True. Exact historical Right-Rear/Right-Front reproduction and remaining draft/numerical differences are shown in rng_pairing_diagnostic.json; a mechanism effect alone is not proof of every historical output. The old error scene and two DEV requests are diagnostics, never final TEST. Eight M3-only replay calls are counted separately from full-M123 calls.
