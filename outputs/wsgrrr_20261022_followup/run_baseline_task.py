"""Run the unchanged current task harness with exact pre-optimization M1/M2."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parent
REPOSITORY=ROOT.parent.parent
snapshot=ROOT/'baseline_source_snapshot'
sys.path.insert(0,str(REPOSITORY));sys.path.insert(0,str(snapshot));os.chdir(snapshot)
import worldstream.m1.state_manager as manager
import worldstream.m2.runtime as runtime
assert Path(manager.__file__).is_relative_to(snapshot)
assert Path(runtime.__file__).is_relative_to(snapshot)
from tools.worldstream_m123.benchmark.worker import main

if __name__=='__main__':main()
