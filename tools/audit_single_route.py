"""Inventory tracked source dependencies before retiring alternative strategies.

This tool writes a reviewable plan only. It never removes files or touches results.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


M2_RETIRED = {'camera_executor.py','camera_feature_cache.py','conditional_prefix.py',
              'source_alignment.py','approximate_probe.py','scheduler.py','cheap_admission.py','execution_policy.py','profiler.py'}
M3_RETIRED = {'carrier_selector.py','batched_executor.py','native_decode.py',
              'numerics.py','projection_precision.py','post_perception.py'}
M1_RETIRED = {'identity_continuity.py','identity_temporal.py','identity_object.py',
              'search_motion.py','correspondence_recovery.py','ego_motion.py',
              'surface_consensus.py','surface_visibility.py','surface_visibility_constraints.py',
              'surface_observability.py','local_surface.py','geometry_adaptive_patch.py',
              'foreground_correspondence.py','observation_refinement.py','measurement_cache.py',
              'bidirectional_geometry.py','dense_support.py','association_tracks.py','world_motion.py',
              'task_facts.py','fact_certifier.py'}
M1_TESTS = {'test_object_surface_identity.py','test_identity_surface_bindings.py','test_identity_surface.py',
    'test_surface_observability.py','test_correspondence_recovery.py','test_identity_metrics.py',
    'test_visible_consensus.py','test_identity_surface.py','test_ego_motion_compensation.py',
    'test_dense_support.py','test_observation_refinement.py','test_quality_coverage.py',
    'test_quality_coverage_metrics.py','test_effective_acceptance.py','test_scene_diagnostics.py',
    'test_final_validation.py','test_trust_refinement.py','test_tools.py','test_independent_contract.py',
    'test_association_tracks.py','test_history_recovery.py','test_task_facts.py','test_fact_certifier.py'}
M2_TESTS = {'test_camera_policy.py','test_camera_executor.py','test_camera_selective_refinement.py',
    'test_source_aligned_conditional.py','test_source_archive.py','test_approximate_probe.py',
    'test_service_contract.py','test_m1_policy.py','test_state_interface_gpu_tools.py',
    'test_refinement_analysis.py','test_cheap_admission.py','test_execution_policy.py'}
M3_KEEP_TESTS = {'__init__.py','conftest.py','test_execution.py','test_sequence_identity.py',
    'test_code_signature.py','test_integration.py','test_prefix_budget_precheck.py','test_trusted_history.py'}


def reason(path):
    name=path.name; text=str(path)
    if text.startswith('tools/worldstream_m2/rearchitecture/'):
        return 'Retire remaining orphan helpers for historical rearchitecture comparisons'
    if text.startswith('tools/system_optimization/'):
        return 'Retire historical multi-strategy comparison; stable sequence checks moved into mainline benchmark'
    if text in ('worldstream/runtime.py','worldstream/execution.py','worldstream/overlap.py'):
        return 'Retire alternative unified/overlap execution route; detector uses the ordered mainline'
    if path.parent==Path('worldstream/m2') and name in M2_RETIRED:
        return 'Retire camera/conditional/approximate M2 executor; window path is the sole algorithm'
    if path.parent==Path('worldstream/m3') and name in M3_RETIRED:
        return 'Retire approximate carriers, alternate decode/batch/seed strategies'
    if path.parent==Path('worldstream/m1') and name in M1_RETIRED:
        return 'Retire experimental identity/geometry/support strategies; retain ordinary A and sparse measured temporal validation'
    if text.startswith('projects/configs/OmniDrive/worldstream_') and name!='worldstream_m123_auto.py':
        return 'Inline the only formal configuration directly on the original OmniDrive baseline'
    if text.startswith('configs/worldstream_') and name!='worldstream_m1_final.yaml':
        return 'Retire historical experimental configuration; keep one formal M1 configuration'
    if text.startswith('tools/worldstream_m2/') and name not in ('__init__.py','common.py','window_validate.py'):
        return 'Retire historical alternate-executor experiment driver'
    if text.startswith('tools/worldstream_m3/'):
        return 'Retire multi-strategy experiment driver; move necessary scoring helpers to benchmark common'
    if text.startswith(('tools/worldstream_m123/stateful_opportunity/','tools/worldstream_m123/large_boundary/')):
        return 'Retire historical comparison driver; move shared audit utilities to benchmark common'
    if text.startswith('tools/worldstream_m1/') and name not in ('__init__.py','dataset.py','io_utils.py',
            'identity_metrics.py','scene_metrics.py','scene_protocol.py'):
        return 'Retire historical experiment driver; retain independent offline scoring utilities'
    if path.parent==Path('tests/worldstream_m1') and name in M1_TESTS:
        return 'Dedicated retired-strategy/configuration test; retain state/transaction/temporal/physical rejection tests'
    if path.parent==Path('tests/worldstream_m2') and name in M2_TESTS:
        return 'Dedicated retired M2 strategy/driver test'
    if path.parent==Path('tests/worldstream_m3') and name not in M3_KEEP_TESTS:
        return 'Dedicated alternate M3 strategy/report test; migrate any shared core assertions before deletion'
    if text in ('tests/worldstream_m123/test_large_boundary.py','tests/worldstream_m123/test_stateful_opportunity.py'):
        return 'Dedicated retired experiment driver test'
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/single_route_convergence_20261022/deletion_plan.json')
    args=parser.parse_args()
    tracked=[Path(p) for p in subprocess.check_output(['git','ls-files','-z'],text=True).split('\0') if p]
    sources=[p for p in tracked if p.suffix in ('.py','.yaml','.md') and p.is_file()
             and not str(p).startswith(('outputs/','data/','ckpts/'))]
    texts={str(p):p.read_text(errors='replace') for p in sources}
    files=[]
    for p in sources:
        if not str(p).startswith(('worldstream/','tools/worldstream_','tests/worldstream_','tools/system_optimization/',
                                  'configs/worldstream_','projects/configs/OmniDrive/worldstream_')):continue
        why=reason(p); module=str(p).removesuffix('.py').replace('/','.')
        callers=[other for other,body in texts.items() if other!=str(p) and
                 (module in body or str(p) in body or
                  (p.suffix=='.py' and f'.{p.stem} import' in body))]
        inheritance=[other for other,body in texts.items() if other.startswith('projects/configs/')
                     and other!=str(p) and p.name in body]
        files.append(dict(path=str(p),action='delete_after_migration' if why else 'keep',
            reason=why or 'Core mainline implementation, correctness/quality validation, or common measurement utility',
            bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            callers=callers,config_dependents=inheritance,
            test_dependents=[x for x in callers if x.startswith('tests/')]))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(scope='tracked source/config/test files only; no outputs/data/weights deletion',
        protected_untracked=['outputs/m3_state_guided_verified_decode/'],files=files),indent=2)+'\n')
    print(json.dumps(dict(files=len(files),planned_deletions=sum(x['action'].startswith('delete') for x in files),output=str(output))))


if __name__=='__main__':main()
