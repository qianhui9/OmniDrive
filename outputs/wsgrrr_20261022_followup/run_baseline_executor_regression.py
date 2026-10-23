"""Same numerical harness, exact Git-baseline executor, no temporal error."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
snapshot=ROOT/'baseline_source_snapshot'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(snapshot));os.chdir(snapshot)
import worldstream.m2.window_executor as executor
assert Path(executor.__file__).is_relative_to(snapshot)
from tools.worldstream_m2.executor_regression import run
run(str(REPO/'ckpts/iter_10548.pth'),str(ROOT/'executor_regression_baseline'))
