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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/module1_source_state_refinement/gpu_validation")
    parser.add_argument("--device", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--zero-cost-followup", action="store_true",
        help="Run fresh regression and controlled zero-update qualification only")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    state = dict(status="preflight", python=PYTHON, cpu_affinity=sorted(os.sched_getaffinity(0)),
        started_utc=datetime.now(timezone.utc).isoformat(), steps=[])
    write(root/"status.json", state)
    try:
        nvitop = subprocess.run(["/home/qh/miniconda3/bin/nvitop", "--once"],
            capture_output=True, text=True, timeout=20)
        (root/"nvitop.txt").write_text(nvitop.stdout+nvitop.stderr)
        state["nvitop_exit_code"] = nvitop.returncode
        selected = wait_idle(root, args.device)
        state["gpu"] = selected
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(selected["index"]),
            MPLCONFIGDIR="/tmp/state_mpl", OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1", OPENCV_FOR_THREADS_NUM="1")
        witness = subprocess.run(["taskset","-c",CPUS,PYTHON,"-c",
            "import torch,json; torch.manual_seed(0); "
            "torch.backends.cuda.matmul.allow_tf32=False; "
            "x=torch.randn(16,16); a=(x.cuda()@x.cuda()).cpu(); b=x@x; "
            "error=float((a-b).abs().max()); assert error<1e-4; "
            "print(json.dumps(dict(cuda=torch.cuda.is_available(),"
            "device=torch.cuda.get_device_name(0),torch=torch.__version__,max_abs=error)))"],
            env=env, capture_output=True, text=True)
        (root/"kernel_witness.txt").write_text(witness.stdout+witness.stderr)
        if witness.returncode:
            raise RuntimeError("Seeded CUDA kernel witness failed")
        # Avoid competing CPU replay/profiling work during timed model runs.
        while not Path('outputs/module1_source_state_refinement/replay/final_ready.json').exists():
            print('Waiting for expanded CPU replay to finish before CUDA timings',flush=True)
            time.sleep(30)
        protocol = Path("outputs/module1_source_state_refinement/replay/protocol.json")
        scenes = json.loads(protocol.read_text())["scenes"]
        common = ["--scenes","2","--repeats",str(args.repeats),"--warmup","1",
                  "--replay-protocol",str(protocol)]
        steps = [
            ("regression", ["-m","pytest","tests/worldstream_m1","tests/worldstream_m2",
                "tests/worldstream_m3","-q","-o","junit_family=legacy",
                "--junitxml="+str(root/"regression_cuda.xml")]),
            ("model_parity", ["-u","-m","tools.worldstream_m1.model_smoke","--compare",
                "--perception-only","--frames","8","--scene",scenes[0]["scene_id"],
                "--m1-config","configs/worldstream_m1_source_state_refined.yaml",
                "--output",str(root/"model_parity")]),
            ("natural", ["-u","-m","tools.worldstream_m2.source_state_refinement_integration",
                "--case","natural","--frames","12",*common,"--output",str(root/"natural")]),
            ("repeat", ["-u","-m","tools.worldstream_m2.source_state_refinement_integration",
                "--case","repeat","--frames","16",*common,"--output",str(root/"repeat")])]
        if args.zero_cost_followup:
            steps = [steps[0], ("repeat", [*steps[-1][1], "--qualify-zero-camera-cost"])]
        steps[-1][1].extend(['--lease-experiment','--selective-probe'])
        state["status"] = "running"
        for name, command in steps:
            wait_idle(root, selected["index"])
            full = ["taskset","-c",CPUS,PYTHON,*command]
            row = dict(name=name,command=full,started_utc=datetime.now(timezone.utc).isoformat())
            state["steps"].append(row)
            write(root/"status.json",state)
            print("Starting",name,flush=True)
            with (root/(name+".log")).open("w") as log:
                result = subprocess.run(full,env=env,stdout=log,stderr=subprocess.STDOUT)
            row.update(exit_code=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
            write(root/"status.json",state)
            if result.returncode:
                raise RuntimeError(name+" failed; inspect "+str(root/(name+".log")))
            if name in ('natural','repeat'):
                quality=json.loads((root/name/'status.json').read_text())
                if quality.get('quality_pass') is not True:
                    raise RuntimeError(name+' failed existing full-task quality checks')
            print("Finished",name,flush=True)
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="blocked" if not state["steps"] else "failed",
            error=repr(exc),finished_utc=datetime.now(timezone.utc).isoformat())
        write(root/"status.json",state)

        raise
    write(root/"status.json",state)



if __name__ == "__main__":
    main()
