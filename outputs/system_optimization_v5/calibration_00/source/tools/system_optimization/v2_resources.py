"""Separate sampled resident-memory controls; never ordinary E2E samples."""
import inspect,json,os
from pathlib import Path
from threading import Event,Thread
from time import perf_counter
from types import SimpleNamespace
import torch
from .audit import write
from .experiment import seed
from .v2_probes import frozen_question


class ResidentSampler:
    def __init__(self,interval=.02):
        self.interval=interval;self.stop=Event();self.rows=[]
    def sample(self):
        rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        status=Path('/proc/self/status').read_text().splitlines()
        threads=int(next(s for s in status if s.startswith('Threads:')).split()[1])
        self.rows.append(dict(elapsed_s=perf_counter()-self.start,rss_bytes=rss,threads=threads))
    def loop(self):
        while not self.stop.wait(self.interval):self.sample()
    def __enter__(self):
        self.start=perf_counter();self.sample();self.thread=Thread(target=self.loop,name='resident-sampler',daemon=True);self.thread.start();return self
    def __exit__(self,*exc):self.stop.set();self.thread.join();self.sample()
    def result(self):
        return dict(interval_s=self.interval,samples=len(self.rows),start_rss_bytes=self.rows[0]['rss_bytes'],
            end_rss_bytes=self.rows[-1]['rss_bytes'],sampled_peak_rss_bytes=max(r['rss_bytes'] for r in self.rows),
            peak_threads_including_sampler=max(r['threads'] for r in self.rows),sampler_threads=1)


@torch.no_grad()
def run(driver,source,dest):
    from mmcv.parallel import collate,scatter
    from tools.worldstream_m3.refinement import freeze_sample_questions
    source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());questions=json.loads((source/'questions.json').read_text())
    indices=protocol['scenes'][0]['indices'][:2];model,dataset=driver.model,driver.dataset
    native=inspect.getclosurevars(driver.native_generate).nonlocals['generate'];sink=SimpleNamespace(ids=[])
    def generate(*a,**kw):
        out=native(*a,**kw);sink.ids.append(out.cpu().tolist());return out
    model.lm_head.generate=generate
    rows=[];references={}
    for mode in ('original','m1','opt_serial','overlap'):
        driver.setup(model,mode,sink,dest,512,10,False)
        for ordinal,index in enumerate(indices):
            seed(index);sink.ids=[];model.save_path=str(dest/f'{mode}_{ordinal}'/'language')+'/'
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            with ResidentSampler() as sampler:
                batch=scatter(collate([freeze_sample_questions(dataset[index],frozen_question(questions[str(index)]))],samples_per_gpu=1),[0])[0]
                seed(index+10000);model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
            if mode=='original':references[index]=sink.ids
            equal=sink.ids==references[index];assert equal,'Resource control changed generation'
            rows.append(dict(mode=mode,index=index,cold=ordinal==0,complete_token_equality=equal,
                gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),
                **sampler.result()))
    model.stream_runtime.close()
    write(dest/'resource_controls.json',dict(scope='Separate actual two-frame same-scene controls, not statistical E2E samples',
        rows=rows,caveat='20ms resident-memory samples may miss shorter spikes; includes shared loaded-model/process memory. One observation sampler thread in every mode. GPU peaks are allocator high-water marks.'))
