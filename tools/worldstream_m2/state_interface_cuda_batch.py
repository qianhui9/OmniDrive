"""Run the authorized local CUDA follow-up, waiting for an idle GPU.

No installation, environment rebuild, remote access or process termination.
All child computations explicitly exclude faulty CPU cores 4 and 5.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CPUS = "0-3,6-31"
PYTHON = "/home/qh/miniconda3/envs/OmniDrive/bin/python"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def query_gpus():
    result = subprocess.run(["nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError("GPU access unavailable in this process: " + result.stderr.strip())
    devices = []
    for row in csv.reader(io.StringIO(result.stdout)):
        index, uuid, name, used, total, util = [x.strip() for x in row]
        devices.append(dict(index=int(index), uuid=uuid, name=name,
            used_mb=int(used), total_mb=int(total), utilization=int(util)))
    processes = subprocess.run(["nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    if processes.returncode:
        raise RuntimeError("Could not inspect GPU processes: " + processes.stderr.strip())
    rows = [[x.strip() for x in row] for row in csv.reader(io.StringIO(processes.stdout))]
    return devices, rows


def wait_idle(root, requested=None):
    while True:
        devices, processes = query_gpus()
        occupied = {row[0] for row in processes if row}
        available = [d for d in devices if d["used_mb"] < 500
            and d["utilization"] == 0 and d["uuid"] not in occupied
            and (requested is None or d["index"] == requested)]
        record = dict(time_utc=datetime.now(timezone.utc).isoformat(),
            devices=devices, processes=processes, available=available)
        with (root/"gpu_availability.jsonl").open("a") as handle:
            handle.write(json.dumps(record)+"\n")
        if available:
            return available[0]
        print("GPU occupied; waiting without stopping any process", flush=True)
        time.sleep(30)
