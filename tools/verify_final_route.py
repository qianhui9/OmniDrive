"""Audit the sole source-linked M1/window M2/exact-prefill M3 route, without weights."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from mmcv import Config
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m2.config import M2Config
from worldstream.m3.config import M3Config
from worldstream.m3.runtime import M3Runtime

ENTRY=Path('projects/configs/OmniDrive/worldstream_m123_auto.py')
ROOT=Path('outputs/single_route_convergence_20261022')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'route_verification')
    parser.add_argument('--tests-log',type=Path,default=ROOT/'tests.log')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    cfg=Config.fromfile(str(ENTRY));baseline=Config.fromfile('projects/configs/OmniDrive/mask_eva_lane_det_vlm.py')
    m1=M1Config.load(cfg.model.m1_config);m2=M2Config(**cfg.model.m2_config);m3=M3Config(**cfg.model.m3_config)
    WorldStateManager(m1);runtime=M3Runtime(m3)
    assert m1.temporal_visibility and m1.local_support and m1.source_linked_state
    assert m2.reuse_boundary=='window' and m2.state_interface=='source_linked' and m2.m1_service=='maintain'
    assert m3.state_context=='augment' and m2.mode==m3.mode=='auto'
    assert list(ENTRY.parent.glob('worldstream_*.py'))==[ENTRY]
    for key in ('img_backbone','pts_bbox_head','map_head','lm_head','tokenizer'):
        if key in baseline.model:assert cfg.model[key]==baseline.model[key],key
    removed=json.loads((ROOT/'deleted_files.json').read_text())
    assert all(not Path(r['path']).exists() and not r['path'].startswith(('outputs/','data/','ckpts/')) for r in removed)
    retired={r['path'][:-3].replace('/','.') for r in removed if r['path'].endswith('.py')}
    sources=[p for folder in ('worldstream','tools','tests','projects/configs/OmniDrive') for p in Path(folder).rglob('*.py')]
    broken=[]
    for p in sources:
        tree=ast.parse(p.read_text());compile(tree,str(p),'exec')
        package=str(p)[:-3].replace('/','.').split('.')[:-1]
        for n in ast.walk(tree):
            names=[]
            if isinstance(n,ast.Import):names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom):
                module=n.module or ''
                if n.level:module='.'.join(package[:len(package)-n.level+1]+([module] if module else []))
                names=[module]+[module+'.'+a.name for a in n.names]
            for name in names:
                if name in retired:broken.append(dict(file=str(p),line=n.lineno,module=name))
    assert not broken,broken
    m3source=Path('worldstream/m3/runtime.py').read_text()
    assert 'lm_head.generate(' not in m3source and 'select_carriers' not in m3source
    m2source=Path('worldstream/m2/runtime.py').read_text()
    assert 'CameraExecutor' not in m2source and 'ConditionalPrefixExecutor' not in m2source
    assert Path('worldstream/m1/state_manager.py').read_text().count('= associate(')==1
    tables=[m2.cost_table,m3.cost_table,m3.quality_certificate,m3.state_quality_certificate]
    log=args.tests_log.read_text();match=re.search(r'(\d+) passed[^\n]*',log)
    passed=match is not None and not any(x in log for x in ('FAILED ','ERROR collecting',' failed,'))
    assert passed,'Regression log incomplete or failed'
    changed=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()
    untouched=[str(p) for p in Path('projects/mmdet3d_plugin/models/dense_heads').glob('*.py')]
    assert not set(changed)&set(untouched),'Original task heads must remain unchanged'
    data=dict(config_loaded=True,one_formal_entry=True,one_m2_executor=True,one_m3_executor=True,
        m1_ordinary_association=True,live_temporal_source_binding=True,baseline_heads_unchanged=True,
        no_imports_to_deleted_modules=True,compiled_files=len(sources),deleted_source_files=len(removed),
        regression_summary=match.group(0),tests_log=str(args.tests_log),tests_log_sha256=sha(args.tests_log),
        M1=asdict(m1),M2=asdict(m2),M3=asdict(m3),
        evidence={p:dict(sha256=sha(p),value=json.loads(Path(p).read_text())) for p in tables},
        verification_scope='Static source/configuration and regression audit; real-model measurements are separate')
    (out/'verification.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (out/'source_manifest.json').write_text(json.dumps(dict(command=['taskset','-c','0-3,6-31',sys.executable,
        '-m','tools.verify_final_route',*sys.argv[1:]],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={str(p):sha(p) for p in sources},formal_config_sha256=sha(ENTRY)),indent=2)+'\n')
    subprocess.run(['git','diff','--check'],check=True)
    print(json.dumps(dict(status='PASS',tests=match.group(0),deleted=len(removed),compiled=len(sources))))


if __name__=='__main__':main()
