"""Fair single-question native HF execution context comparison, same weights.

Inference-mode removes autograd view/version bookkeeping; it is not a new
decoder kernel and no gain is attributed to world state. No backend upgrade.
"""
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import torch
from .audit import write, stats
from worldstream.execution import FrameIdentity, FrameExecutionRecord, StageRecorder


@torch.no_grad()
def run(model, native_generate, inputs, dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    versions=[(n,p._version,p.dtype) for n,p in model.named_parameters()]
    rows=[];reference=None
    for rep in range(7):
        for name in (('no_grad','inference_mode') if rep%2 else ('inference_mode','no_grad')):
            torch.manual_seed(123);torch.cuda.manual_seed_all(123)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            tick=perf_counter()
            with torch.inference_mode() if name=='inference_mode' else nullcontext():
                output=native_generate(inputs=inputs['ids'],images=inputs['images'],do_sample=True,
                    temperature=.1,top_p=.75,num_beams=1,max_new_tokens=320,use_cache=True)
            torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
            ids=output.detach().cpu().tolist()
            if name=='no_grad':reference=ids
            rows.append(dict(method=name,repeat=rep,warmup=rep<2,wall_ms=wall,ids=ids,
                generated_tokens=output.shape[-1]-1,peak_allocated=torch.cuda.max_memory_allocated()))
    # Compare each paired complete output, including EOS and unchanged sampling.
    for row in rows:
        row['exact_token_equality']=row['ids']==next(r['ids'] for r in rows if r['repeat']==row['repeat'] and r['method']=='no_grad')
    unchanged=versions==[(n,p._version,p.dtype) for n,p in model.named_parameters()]
    write(dest/'results.json',dict(rows=rows,weights_unchanged=unchanged,
        summary={method:dict(wall_ms=stats([r['wall_ms'] for r in rows if not r['warmup'] and r['method']==method]),
            exact_token_equality=all(r['exact_token_equality'] for r in rows if r['method']==method)) for method in ('no_grad','inference_mode')},
        scope='One real planning prompt/carrier; 2 warmup + 5 measured paired native complete generations; no state mechanism',
        decision='diagnostic_only_single_input_not_a_deployment_certificate'))
    assert unchanged
