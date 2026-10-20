"""Offline read-only evaluation of completed, independently committed runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run(root,cohort):
    root=Path(root);variants=('baseline','old_b','a','b','c','aligned') if cohort=='fixed' else ('baseline','c','aligned')
    remaining=list(variants)
    while remaining:
        ready=[v for v in remaining if (root/cohort/v/'summary.json').is_file()]
        if not ready:time.sleep(5);continue
        for variant in ready:
            source=root/cohort/variant;out=root/(cohort+'_evaluation')/variant;out.mkdir(parents=True,exist_ok=False)
            summary=json.loads((source/'summary.json').read_text())
            if not summary['reproduced'] or not summary['source_unchanged']:raise ValueError('Online replay did not validate')
            jobs=[('history_recovery_metrics','funnel',True),('motion_scoring','correspondence',True),
                  ('offline_identity_scoring','identity',True),('scene_revalidate','state_ledger.json',False),
                  ('motion_revalidate','depth_ledger.json',False)]
            for module,target,is_dir in jobs:
                command=[sys.executable,'-m','tools.worldstream_m1.'+module,'--root',str(source),'--output',str(out/target)]
                start=time.time()
                with (out/(target.replace('.json','')+'.stdout.log')).open('w') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                receipt=dict(command=command,returncode=result.returncode,wall_seconds=time.time()-start)
                (out/(target.replace('.json','')+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
                if result.returncode:raise RuntimeError(str(receipt))
            remaining.remove(variant);print('Evaluated',cohort,variant,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--cohort',choices=['fixed','expanded'],required=True)
    a=p.parse_args();run(a.root,a.cohort)


if __name__=='__main__':main()
