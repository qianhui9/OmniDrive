"""Finish evaluation after the existing GPU worker exits; no Git mutation here."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .benchmark.common import sha256,write
from .frozen_main_benchmark import ROOT,AUTHORITY,read,validate_frozen


def finalize(root):
    assert not {4,5}&os.sched_getaffinity(0)
    state=root/'pipeline_progress.json'
    def progress(stage,**extra):
        write(state,dict(stage=stage,UTC=datetime.now(timezone.utc).isoformat(),**extra))
        print(stage,flush=True)
    def run(module,*args):
        command=[sys.executable,'-m',module,*map(str,args)]
        subprocess.run(command,check=True)
    while not (root/'main/run_status.json').exists():
        if (root/'main_failure.json').exists():
            raise RuntimeError('Main real GPU worker failed; preserve all evidence and do not synthesize results')
        progress('WAITING_FOR_EXISTING_MAIN_GPU',progress=read(root/'main/progress.json') if (root/'main/progress.json').exists() else None)
        time.sleep(30)
    assert read(root/'main/run_status.json')['status']=='COMPLETED'
    progress('VERIFYING_MAIN_RECEIPTS')
    run('tools.worldstream_m123.frozen_main_report','--root',root,'--verify')
    progress('ASSEMBLING_MAIN_REPORT_AND_OFFICIAL_LANE_SUBSET')
    run('tools.worldstream_m123.frozen_main_report','--root',root)
    progress('FINAL_CPU_REGRESSION')
    run('tools.worldstream_m123.answer_validity_validate','--output',root/'cpu_validation')
    progress('FINAL_RTX3090_CUDA_REGRESSION')
    run('tools.worldstream_m123.answer_validity_validate','--gpu','--output',root/'cuda_validation')
    comparisons=('m1_metrics.json','m2_metrics.json','m3_metrics.json','main_results.json',
        'workload_results.json','temporal_results.json','condition_results.json','task_results.json',
        'position_error_breakdown.json','severe_error_cases.json','map_quality_metrics.json','final_research_verdict.json')
    originals={name:read(root/name) for name in comparisons}
    write(root/'pre_archive_metric_hashes.json',{name:sha256(root/name) for name in comparisons})
    progress('LOSSLESS_CURRENT_RUN_RECEIPT_ARCHIVAL')
    run('tools.worldstream_m123.frozen_main_archive','--root',root)
    progress('VERIFYING_ARCHIVED_PAIRS')
    run('tools.worldstream_m123.frozen_main_report','--root',root,'--verify')
    progress('RECOMPUTING_ALL_RESULTS_FROM_ARCHIVED_RAW_RECEIPTS')
    run('tools.worldstream_m123.frozen_main_report','--root',root)
    same={name:read(root/name)==value for name,value in originals.items()}
    assert all(same.values()),same
    write(root/'reanalysis_validation.json',dict(status='PASS',identical_scientific_metrics=same,
        inputs='Lossless exact-byte raw receipt chunks via chronological index',GPU_inference_calls=0))
    progress('FINAL_SOURCE_HISTORY_WEIGHT_VALIDATION')
    validate_frozen(root)
    for name,h in read(root/'historical_output_hashes.json').items():assert sha256(name)==h,name
    assets=read(AUTHORITY/'freeze/protected_checkpoints.json')
    for name,entry in assets.items():
        assert sha256(name)==entry['sha256'],name
    write(root/'weight_validation.json',dict(status='PASS',assets=len(assets),all_original_bytes_unchanged=True,
        checksums_source=str(AUTHORITY/'freeze/protected_checkpoints.json')))
    cpu=read(root/'cpu_validation/summary.json');cuda=read(root/'cuda_validation/summary.json')
    assert cpu['exit_code']==cuda['exit_code']==0 and cuda['CUDA_available']
    main=read(root/'main/run_status.json');pilot=read(root/'pilot/run_status.json')
    names=('frozen_main_config.json','frozen_main_config.sha256','baseline_source_audit.json','experiment_protocol.json',
        'data_role_manifest.json','workload_manifest.json','scene_condition_manifest.json','paired_gpu_receipts.jsonl',
        'per_frame_metrics.jsonl','per_question_metrics.jsonl','m1_metrics.json','m2_metrics.json','m3_metrics.json',
        'e2e_performance_report.md','quality_evaluation_report.md','workload_comparison.md','condition_comparison.md',
        'module_efficiency_report.md','main_results_tables.md','final_research_verdict.json','REPRODUCE.md')
    assert all((root/name).is_file() for name in names)
    write(root/'required_deliverables_manifest.json',{name:dict(bytes=(root/name).stat().st_size,sha256=sha256(root/name)) for name in names})
    write(root/'completion.json',dict(status='READY_FOR_GIT_REVIEW',main_full_model_calls=main['full_model_calls'],
        pilot_full_model_calls=pilot['full_model_calls'],total_real_full_model_calls=main['full_model_calls']+pilot['full_model_calls'],
        historical_receipts_reused_as_new=0,CPU=cpu['counts'],CUDA=cuda['counts'],
        actual_CUDA_hardware=cuda['device'],source_config_history_and_weights='PASS',archive_reanalysis='PASS',
        production_default=False,independent_TEST=False,git_mutated_by_pipeline=False))
    progress('READY_FOR_GIT_REVIEW')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,default=ROOT);a=parser.parse_args()
    with (a.root/'finalization_execution.log').open('x') as log:
        from contextlib import redirect_stdout,redirect_stderr
        with redirect_stdout(log),redirect_stderr(log):
            try:finalize(a.root)
            except BaseException:
                write(a.root/'finalization_failure.json',dict(status='FAILED_VALIDATION',error=traceback.format_exc()))
                traceback.print_exc();raise


if __name__=='__main__':main()
