"""Controlled deployed-weight EVA block 0 on RTX 3090. Medians of 20 executions per shape, after five warmups for Full/Packed and three for Naive; one seeded changed-window mask per sparse count. The shape-invariant dense reference is measured once (20 executions) and repeated across x. Intervals include packing, cache copy, scatter and launch gaps."""
from tools.worldstream_m2.plot_results import draw
if __name__ == "__main__": draw("c")
