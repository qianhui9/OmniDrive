"""Three alternating real-service pairs; no runtime hooks or policy overrides."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def run(root,prior,rounds):
    root=Path(root);prior=Path(prior);root.mkdir(parents=True,exist_ok=True)
    sequence=json.loads((prior/'sequence40.json').read_text())
    questions=[json.loads(x) for x in (prior/'formal40/per_question_metrics.jsonl').read_text().splitlines()]
    sequence['questions']={q['frame_id']:[q['question']] for q in questions}
    if len(sequence['tokens'])!=40 or len(sequence['questions'])!=40:raise ValueError('Need the complete original 40-question sequence')
    sequence_file=root/'service_sequence.json';sequence_file.write_text(json.dumps(sequence,indent=2))
    commands=[]
    for repeat in range(rounds):
        for name,condition in ([('native','B0'),('formal','B3')] if repeat%2==0 else [('formal','B3'),('native','B0')]):
            dest=root/('r'+str(repeat)+'_'+name)
            command=['taskset','-c','0-3,6-31',sys.executable,'-u','-m','tools.worldstream_m123.benchmark.worker',
                '--output',str(dest),'--run-root',str(root),'--condition',condition,'--workload','language',
                '--limit','40','--warmup','1','--seed','0','--repeat',str(repeat),'--measurement-set','efficiency',
                '--sequence-file',str(sequence_file)]
            if name=='formal':command+=['--m2-acceptance']
            log=root/(dest.name+'.log')
            record=dict(name=dest.name,command=command,log=str(log),start=datetime.now(timezone.utc).isoformat())
            commands.append(record)
            (root/'service_commands.json').write_text(json.dumps(commands,indent=2))
            print('START',dest.name,flush=True)
            with log.open('w') as handle:result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode,end=datetime.now(timezone.utc).isoformat())
            (root/'service_commands.json').write_text(json.dumps(commands,indent=2))
            if result.returncode:raise RuntimeError('Service run failed: '+str(log))
            print('DONE',dest.name,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--prior',required=True)
    parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args();run(args.root,args.prior,args.rounds)
