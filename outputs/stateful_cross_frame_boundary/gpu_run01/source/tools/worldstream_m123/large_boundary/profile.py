"""Temporary scopes and non-overlapping kernel attribution for torch 1.13."""
from collections import Counter, defaultdict
from contextlib import nullcontext
from functools import wraps
import json
from time import perf_counter
import torch
from tools.worldstream_m123.stateful_opportunity.common import replace_method


def compute_activity(name):
    return not any(word in name.lower() for word in ('memcpy','memset'))


def stage_for(scopes):
    for name in ('eva','detection','map','llm_prefill','llm_decode','m1_precheck',
                 'm1_commit','input_adapter','output_adapter','perception_position'):
        if name in scopes:
            return name
    return 'llm_sampling_and_preparation' if 'llm_generation' in scopes else 'other_model'


def aggregate_events(events):
    """Each CPU-attached CUDA activity is attributed once to its owning stage.

    Inclusive module scopes are retained separately and must never be summed
    across parents and children. Validate attribution against real device events.
    """
    stages = defaultdict(lambda:dict(kernel_ms=0.,transfer_ms=0.,kernel_count=0,activity_count=0))
    scopes_total = defaultdict(float)
    names = defaultdict(lambda:[0.,0])
    device_total = device_compute = attached_total = 0.
    device_count = attached_count = 0
    for event in events:
        if event.device_type == torch.autograd.DeviceType.CUDA:
            duration = event.time_range.elapsed_us()/1000
            device_total += duration; device_count += 1
            if compute_activity(event.name): device_compute += duration
            continue
        if not event.kernels:
            continue
        ancestors=[]; cursor=event
        while cursor is not None:
            if cursor.name.startswith('lb::'): ancestors.append(cursor.name[4:])
            cursor=cursor.cpu_parent
        stage=stage_for(ancestors)
        for kernel in event.kernels:
            duration=kernel.duration/1000
            attached_total+=duration; attached_count+=1
            row=stages[stage]; row['activity_count']+=1
            if compute_activity(kernel.name):
                row['kernel_ms']+=duration; row['kernel_count']+=1
                names[kernel.name][0]+=duration; names[kernel.name][1]+=1
                for scope in set(ancestors): scopes_total[stage+'|'+scope]+=duration
            else:
                row['transfer_ms']+=duration
    valid = abs(attached_total-device_total) <= max(.01,device_total*.001) and attached_count==device_count
    return dict(attribution_valid=valid, device_activity_ms=device_total,
        device_compute_kernel_ms=device_compute, device_activity_count=device_count,
        attached_activity_ms=attached_total, attached_activity_count=attached_count,
        stages=dict(stages), inclusive_scope_kernel_ms=dict(scopes_total),
        top_kernels=[dict(name=n,kernel_ms=v[0],calls=v[1]) for n,v in
            sorted(names.items(),key=lambda item:-item[1][0])[:40]],
        interpretation='Stage kernel sums are disjoint. Inclusive scopes overlap; CUDA event intervals include host launch gaps and are reported separately.')


def aggregate_trace(path, reference=None):
    """Authoritative device activities -> unique CUDA runtime correlation -> scopes.

    Torch 1.13 CPU FunctionEvent.kernels can duplicate/drop attachments. Do not
    rescale those totals: use the preserved Kineto trace correlation identifiers.
    """
    with open(path) as handle:events=json.load(handle)['traceEvents']
    ranges=defaultdict(list);runtime=defaultdict(list)
    for e in events:
        if e.get('ph')!='X':continue
        if e.get('cat')=='user_annotation' and e['name'].startswith('lb::'):
            ranges[e['tid']].append(e)
        elif e.get('cat')=='cuda_runtime' and 'correlation' in e.get('args',{}):
            runtime[e['tid']].append(e)
    correlations={};ambiguous=set()
    for tid,rows in runtime.items():
        scopes=sorted(ranges[tid],key=lambda e:e['ts']);cursor=0;active=[]
        for e in sorted(rows,key=lambda e:e['ts']):
            while cursor<len(scopes) and scopes[cursor]['ts']<=e['ts']:
                active.append(scopes[cursor]);cursor+=1
            active=[s for s in active if s['ts']+s['dur']>=e['ts']]
            labels=[s['name'][4:] for s in active]
            key=e['args']['correlation']
            value=(stage_for(labels),tuple(labels))
            if key in correlations and correlations[key]!=value:ambiguous.add(key)
            correlations[key]=value
    stages=defaultdict(lambda:dict(kernel_ms=0.,transfer_ms=0.,kernel_count=0,activity_count=0,transfer_bytes=0))
    inclusive=defaultdict(float);names=defaultdict(lambda:[0.,0]);missing=[]
    total=compute=0.;count=0;compute_count=0
    for e in events:
        cat=e.get('cat')
        if e.get('ph')!='X' or cat not in ('kernel','gpu_memcpy','gpu_memset'):continue
        count+=1;duration=e['dur']/1000;total+=duration
        key=e.get('args',{}).get('correlation')
        if key not in correlations or key in ambiguous:
            missing.append(dict(name=e['name'],correlation=key,duration_ms=duration));stage,labels='unattributed',()
        else:stage,labels=correlations[key]
        row=stages[stage];row['activity_count']+=1
        if cat=='kernel':
            compute+=duration;compute_count+=1;row['kernel_ms']+=duration;row['kernel_count']+=1
            names[e['name']][0]+=duration;names[e['name']][1]+=1
            for label in set(labels):inclusive[stage+'|'+label]+=duration
        else:
            row['transfer_ms']+=duration;row['transfer_bytes']+=e.get('args',{}).get('bytes',0)
    agrees=(reference is None or (count==reference['device_activity_count'] and
        abs(total-reference['device_activity_ms'])<=max(.01,total*.001)))
    accounting=(agrees and sum(r['activity_count'] for r in stages.values())==count and
        abs(sum(r['kernel_ms']+r['transfer_ms'] for r in stages.values())-total)<=max(.001,total*1e-8))
    return dict(attribution_valid=not missing and agrees,method='Unique trace device activity + CUDA runtime correlation + containing CPU scope',
        accounting_valid=accounting,attribution_complete=not missing,
        unattributed_activity_ms=sum(r['duration_ms'] for r in missing),
        attributed_time_fraction=1-sum(r['duration_ms'] for r in missing)/total if total else None,
        device_activity_ms=total,device_compute_kernel_ms=compute,device_activity_count=count,
        compute_kernel_count=compute_count,stages=dict(stages),inclusive_scope_kernel_ms=dict(inclusive),
        top_kernels=[dict(name=n,kernel_ms=v[0],calls=v[1]) for n,v in sorted(names.items(),key=lambda item:-item[1][0])[:40]],
        missing_correlations=missing[:30],missing_correlation_count=len(missing),
        independently_matches_profiler_device_events=agrees,
        legacy_cpu_attachment_consistent=reference.get('attribution_valid') if reference else None,
        interpretation='Each trace device activity counted once; no scaling or inferred kernels. Inclusive scopes overlap. CUDA intervals and CPU wall stay separate.')


class Scopes:
    def __init__(self, stack, ranges=True):
        self.stack=stack; self.events=[]; self.calls=Counter(); self.llm_steps=[]
        self.ranges=ranges;self.seen=set()

    def wrap(self, obj, attribute, label):
        if (id(obj),attribute) in self.seen:return
        self.seen.add((id(obj),attribute))
        original=getattr(obj,attribute)
        @wraps(original)
        def method(*a,**kw):
            name=label(a,kw) if callable(label) else label
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); tick=perf_counter()
            with torch.autograd.profiler.record_function('lb::'+name) if self.ranges else nullcontext():
                result=original(*a,**kw)
            end.record(); elapsed=(perf_counter()-tick)*1000
            self.events.append((name,start,end,elapsed)); self.calls[name]+=1
            if name in ('llm_prefill','llm_decode'):
                value=kw.get('inputs_embeds')
                if value is None: value=kw.get('input_ids')
                self.llm_steps.append(dict(stage=name,input_shape=list(value.shape),
                    past_tokens=kw['past_key_values'][0][0].shape[-2] if kw.get('past_key_values') else 0,
                    cpu_enqueue_ms=elapsed,event_index=len(self.events)-1))
            return result
        self.stack.enter_context(replace_method(obj,attribute,method))

    def collect(self):
        rows=defaultdict(lambda:dict(cuda_event_interval_ms=0.,cpu_enqueue_wall_ms=0.,calls=0))
        for label,a,b,cpu in self.events:
            r=rows[label];r['cuda_event_interval_ms']+=a.elapsed_time(b)
            r['cpu_enqueue_wall_ms']+=cpu;r['calls']+=1
        steps=[]
        for row in self.llm_steps:
            row=dict(row); _,a,b,_=self.events[row.pop('event_index')]
            row['cuda_event_interval_ms']=a.elapsed_time(b);steps.append(row)
        self.events.clear();self.llm_steps.clear();self.calls.clear()
        return dict(inclusive_module_intervals=dict(rows),llm_forward_steps=steps,
            additivity='Nested intervals overlap; never sum module P50s or CPU enqueue time with GPU intervals.')


def install_scopes(model, stack):
    from worldstream.m3.visual_state import base_model
    import worldstream.m1.omnidrive_adapter as adapter
    scopes=Scopes(stack)
    def module(obj,label):scopes.wrap(obj,'forward',label)
    module(model.img_backbone,'eva')
    for i,block in enumerate(model.img_backbone.blocks):
        module(block,f'eva/block_{i}');module(block.attn,f'eva/attention_{i}');module(block.mlp,f'eva/ffn_{i}')
    for name,head in [('detection',model.pts_bbox_head),('map',model.map_head)]:
        module(head,name)
        for attribute in ('pre_update_memory','temporal_alignment','post_update_memory'):
            scopes.wrap(head,attribute,name+'/'+attribute)
        for i,layer in enumerate(head.transformer.query_decoder._layers):
            module(layer.transformer_layers[0],f'{name}/self_attention_{i}')
            module(layer.transformer_layers[2],f'{name}/cross_attention_{i}')
            module(layer.transformer_layers[4],f'{name}/ffn_{i}')
        for attribute in ('input_projection','output_projection','query_pos','time_embedding','ego_pose_pe'):
            value=getattr(head,attribute,None)
            if isinstance(value,torch.nn.Module):module(value,('head_shared' if attribute in ('query_pos','time_embedding','ego_pose_pe') else name)+'/'+attribute)
    scopes.wrap(model,'position_embeding','perception_position')
    scopes.wrap(adapter,'frame_from_model','input_adapter');scopes.wrap(adapter,'outputs_from_model','output_adapter')
    scopes.wrap(model.m1_manager,'precheck_with_observation','m1_precheck')
    scopes.wrap(model.m1_manager,'commit','m1_commit')
    base=base_model(model.lm_head)
    module(base,lambda a,k:'llm_prefill' if k.get('inputs_embeds') is not None and k['inputs_embeds'].shape[1]>1 else 'llm_decode')
    module(base.lm_head,'llm/output_projection')
    for i,layer in enumerate(base.model.layers):
        module(layer.self_attn,f'llm/attention_{i}');module(layer.mlp,f'llm/ffn_{i}')
    scopes.wrap(model.lm_head,'generate','llm_generation')
    return scopes
