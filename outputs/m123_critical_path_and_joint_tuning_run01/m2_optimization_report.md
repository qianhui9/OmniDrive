# M2 executor stop decision

No serving M2 implementation changed. P0 verified real selective QKV/attention/MLP execution, small gather/scatter/cache costs and dominant necessary global suffix computation. An executor rewrite would add complexity for small remaining overhead. Full/Incremental is an algorithmic reuse ablation, not a before/after code-optimization speedup. Subsequent changes to existing refresh parameters are reported as parameter effects. Source hashes preserve the unchanged M2.
