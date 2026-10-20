"""Explicit cost gate. Native-prefix probes remain an exploratory upper bound."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from tools.worldstream_m1.io_utils import write_json


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def summarize(root):
    requests=read_rows(root/'phase1/requests.jsonl');probes=read_rows(root/'phase1/probes.jsonl')
    groups=defaultdict(list)
    for p in probes:groups[(p['population'],p['task'])].append(p)
    costs=[];initial=[]
    for (population,task),rows in groups.items():
        for k in (1,2,4,8,16):
            measured=[c for r in rows for c in r['costs'] if c['k']==k and not c['warmup']]
            if not measured:continue
            serial=float(np.median([c['serial']['wall_ms']/k for c in measured]))
            verify=float(np.median([c['parallel']['wall_ms'] for c in measured]))
            kv=float(np.median([c['owned_branch_copy_ms'] for c in measured]))
            expectations=[];processing=[]
            for r in rows:
                for block in r['shadow']:
                    pp=block['probabilities'][:k]
                    expectations.append(float(np.cumprod(pp).sum()))
                    processing.append(block['processing_wall_ms']*len(pp)/len(block['probabilities']))
            expected=float(np.mean(expectations))
            process=float(np.mean(processing))
            initial_expected=float(np.mean([v['expected_accepted'] for r in rows for v in r['initial'] if v['k']==k]))
            costs.append(dict(population=population,task=task,k=k,historical_requests=len(rows),
                native_prefix_positions=len(expectations),serial_token_wall_ms=serial,verify_wall_ms=verify,
                owned_kv_copy_wall_ms=kv,logits_processing_wall_ms=process,
                expected_accepted_per_verification=expected,initial_expected_accepted=initial_expected,
                optimistic_net_saved_ms=expected*serial-verify,
                net_with_owned_kv_and_processing_ms=expected*serial-verify-kv-process,
                assumptions='native-prefix off-policy expectation, no real runtime savings; no sampling/correction cost yet'))
    census=dict(total_requests=len(requests),historical_draft_available=sum(r['historical_draft_available'] for r in requests),
        availability_fraction=sum(r['historical_draft_available'] for r in requests)/max(1,len(requests)),
        by_population={p:dict(requests=sum(r['population']==p for r in requests),
            available=sum(r['population']==p and r['historical_draft_available'] for r in requests))
            for p in sorted({r['population'] for r in requests})},
        by_task=dict(Counter(r['task'] for r in requests)),
        population='exploratory_reused_population',independent_heldout=False,
        source='phase1/requests.jsonl; all completed requests including cold starts and misses')
    write_json(root/'historical_draft_census.json',census)
    write_json(root/'draft_acceptance_analysis.json',dict(groups=[dict(population=p,task=t,requests=len(rs),
        lcp=[r['lcp'] for r in rs],gaps_s=[r['gap_s'] for r in rs],initial=[r['initial'] for r in rs])
        for (p,t),rs in groups.items()],source='phase1/probes.jsonl and phase1/teacher_forced/',
        warning='LCP is diagnostic only; target-prefix probabilities are offline, not an online proposal source'))
    write_json(root/'verification_gpu_profile.json',dict(rows=[dict(key=r['key'],measurements=r['costs']) for r in probes],
        source='Fresh full original LLaMA, actual complete current visual prompts, paired k serial and k parallel forwards'))
    write_json(root/'decode_cost_benefit.json',dict(rows=costs,
        equation='E[accepted]*serial_token_time - parallel_verification - KV - processing - proposal - correction',
        note='First candidate uses prompt logits; k input tokens provide k subsequent logits including a bonus token. '
             'The correction/bonus still needs one native forward before the next block. '
             'Each emitted token other than the first costs one native forward at baseline.'))
    native=[c for c in costs if c['population'] in ('planning','native_multi')]
    promising=[c for c in native if c['net_with_owned_kv_and_processing_ms']>0]
    completed=(root/'phase1/status.json').exists()
    result=dict(phase1_completed=completed,
        research_opportunity=bool(promising) and completed,production_go=False,independent_population=False,
        profitable_candidates=promising,
        decision='incomplete_no_runtime_admission' if not completed else 'research_prototype_only' if promising else 'no_go_stop_generation_loop_work',
        reason='Only exploratory native-prefix evidence; actual runtime and frozen task quality still required' if promising else
               'No fixed draft length has positive measured native-prefix expected savings after ownership and processing costs')
    write_json(root/'phase1/gate.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/m3_state_guided_verified_decode')
    args=p.parse_args();result=summarize(Path(args.output));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
