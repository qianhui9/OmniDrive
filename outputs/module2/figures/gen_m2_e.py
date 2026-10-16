"""EVA block 0 on RTX 3090: component-wise medians over 20 instrumented executions for each count. The stack sums marginal component medians and need not equal the median complete wall time. Host/synchronization residual is wall minus event intervals, not an independent device kernel. Hatching distinguishes components in grayscale."""
from tools.worldstream_m2.plot_results import draw
if __name__ == "__main__": draw("e")
