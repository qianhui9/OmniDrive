"""Verify the sole route without loading weights or running GPU inference."""
import ast
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from mmcv import Config
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m2.config import M2Config
from worldstream.m3.config import M3Config
from worldstream.m3.runtime import M3Runtime

ROOT=Path('outputs/final_route')
ENTRY=Path('projects/configs/OmniDrive/worldstream_m123_auto.py')
OUTPUT=ROOT


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):(OUTPUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    parser.add_argument('--tests-log',type=Path,default=ROOT/'tests.log')
    args=parser.parse_args();OUTPUT=args.output_dir;OUTPUT.mkdir(parents=True,exist_ok=True)
    cfg=Config.fromfile(str(ENTRY));m1=M1Config.load(cfg.model.m1_config)
    m2=M2Config(**cfg.model.m2_config);m3=M3Config(**cfg.model.m3_config)
    WorldStateManager(m1);runtime=M3Runtime(m3)
    assert m1.dqtrack_mode=='off' and m2.mode==m3.mode=='auto'
    assert not m3.offline_experiment and m3.state_context=='none' and not m3.batched_decode
    assert not m2.experimental_conditional_reuse and not m2.experimental_exact_lease_separation
    chain=[]
    def visit(path):
        if str(path) in chain:return
        chain.append(str(path));tree=ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_base_' for t in node.targets):
                bases=ast.literal_eval(node.value)
                for base in ([bases] if isinstance(bases,str) else bases):visit(path.parent/base)
    visit(ENTRY)
    artifacts=json.loads((ROOT/'artifact_manifest.json').read_text())
    assert all(sha(r['path'])==r['sha256'] for r in artifacts)
    source=Path('worldstream/m1/state_manager.py').read_text()
    assert source.count('pairs, costs = associate(')==1 and 'from .dqtrack_bridge' not in source
    expected=json.loads((ROOT/'preserved_source_hashes.json').read_text())
    assert all(sha(p)==digest for p,digest in expected.items())
    table=json.loads(Path(m2.cost_table if m2.reuse_boundary=='window' else m2.camera_cost_table).read_text())
    approved=[dict(count=r.get('updated',r.get('changed_cameras')),key=r['key']) for r in table['records'] if r.get('quality_certified') is True]
    assert not runtime.scheduler.costs.get('approved') and not runtime.scheduler.quality.get('approved')
    configs={path:sha(path) for path in chain+[cfg.model.m1_config]}
    save('config_manifest.json',dict(entry=str(ENTRY),inheritance=chain,sha256=configs,M1=asdict(m1),M2=asdict(m2),M3=asdict(m3),artifacts=artifacts,
        M2_active_cost_table=dict(path=m2.cost_table if m2.reuse_boundary=='window' else m2.camera_cost_table,
            sha256=sha(m2.cost_table if m2.reuse_boundary=='window' else m2.camera_cost_table)),
        M2_approved_profiles=approved,M3_approved_rows=0))
    paths=[p for p in Path('worldstream').rglob('*.py')]+[p for p in Path('projects/mmdet3d_plugin/models').rglob('*.py')]+[Path(__file__)]+[Path(p) for p in configs]
    save('source_manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        worktree='uncommitted local changes for review; no commit/push',sha256={str(p):sha(p) for p in paths},preserved_critical_sources=expected,
        python=sys.executable,command=['taskset','-c','0-3,6-31',sys.executable,'-m','tools.verify_final_route',*sys.argv[1:]]))
    plan=json.loads((ROOT/'cleanup_plan.json').read_text());deleted=[p for p in plan['delete'] if not Path(p).exists()]
    protected=json.loads((ROOT/'initial_workspace.json').read_text())['protected_files']
    protected.update({r['path']:dict(bytes=r['bytes']) for r in json.loads((ROOT/'protected_ignored_files.json').read_text())})
    assert all(Path(p).is_file() and Path(p).stat().st_size==v['bytes'] and ('mtime_ns' not in v or Path(p).stat().st_mtime_ns==v['mtime_ns']) for p,v in protected.items())
    # Resolve actual Python imports to deleted repository modules. Generic words
    # in comments and historical output names are not import dependencies.
    removed={p[:-3].replace('/','.') for p in deleted if p.endswith('.py') and not p.startswith('outputs/')}
    broken=[];compiled=0
    for directory in ('worldstream','tools','tests','projects/configs/OmniDrive'):
        for p in Path(directory).rglob('*.py'):
            tree=ast.parse(p.read_text());compile(tree,str(p),'exec');compiled+=1
            package=str(p)[:-3].replace('/','.').split('.')[:-1]
            for node in ast.walk(tree):
                names=[]
                if isinstance(node,ast.Import):names=[x.name for x in node.names]
                if isinstance(node,ast.ImportFrom):
                    module=node.module or ''
                    if node.level:module='.'.join(package[:len(package)-node.level+1]+([module] if module else []))
                    names=[module]+[module+'.'+x.name for x in node.names]
                for name in names:
                    if name in removed:broken.append(dict(file=str(p),line=node.lineno,module=name))
    assert not broken,broken
    log=args.tests_log.read_text();match=re.search(r'(\d+) passed.* in ([^\n]+)',log)
    passed=bool(match) and 'FAILED ' not in log and 'ERROR collecting' not in log
    save('verification.json',dict(config_loaded=True,runtime_initialized=True,tests_passed=passed,
        verification_scope='static configuration/source audit; runtime initialized without weights',
        tests_summary=match.group(0) if match else 'INCOMPLETE_OR_FAILED',
        tests_log=str(args.tests_log),tests_log_sha256=sha(args.tests_log),
        artifact_hashes_preserved=True,baseline_heads_and_sources_unchanged=True,protected_files_unchanged=True,
        sole_identity_assignment=True,no_DQTrack_imports=True,compiled_files=compiled,deleted_module_imports=broken,
        M2_M3_modes='auto',final_integrated_inference='NOT_RUN',performance='NOT_MEASURED',weights_loaded=False,
        cleanup_deleted_files=len(deleted),protected_untracked_files=len(protected),
        output_exceptions='Preexisting untracked directory and ignored files preserved by explicit user instruction'))
    save('summary.json',dict(route=str(ENTRY),M1='ordinary A, DQTrack off, original unified allocator and transactions',
        M2=f'{m2.reuse_boundary} Auto; only keyed quality/cost approvals; otherwise native EVA',
        M3='Auto with unchanged empty approved evidence; native generation fallback',
        final_integrated_results='NOT_RUN',performance='NOT_MEASURED',tests=match.group(0) if match else 'INCOMPLETE_OR_FAILED',
        cleanup_deleted_files=len(deleted),historical_evidence='historical_selection_evidence.json; not final-route measurement'))
    subprocess.run(['git','diff','--check'],check=True)
    print(json.dumps(dict(config_loaded=True,compiled=compiled,tests=match.group(0) if match else 'INCOMPLETE',deleted=len(deleted)),ensure_ascii=False))
    if not passed:raise SystemExit('Tests incomplete or failed; see tests.log')


if __name__=='__main__':main()
