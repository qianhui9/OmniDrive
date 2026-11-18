"""Per-trial statistics first; sample SD across independent trials (ddof=1)."""
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from tools.environment_stability_eval.common import csv_write, jsonl, read, write

METRICS = ('cpu_host_ms', 'gpu_device_ms', 'component_wall_ms')
COMPONENTS = {
    'OGSM.Precheck': ('frame', False), 'OGSM.Commit': ('frame', False),
    'OGSM.Total': ('frame', False), 'DAVP.Total': ('frame', True),
    'DAVP.Patch Embedding': ('frame', True), 'DAVP.Visual Planning': ('frame', True),
    'DAVP.Incremental EVA': ('frame', True), 'DAVP.Global Refresh': ('frame', True),
    'DAVP.Global Residual Reuse': ('frame', True),
    'Detection Head': ('frame', True), 'Mapping Head': ('frame', True),
    'Original Perception': ('frame', True), 'QCSE.Total': ('frame', True),
    'QCSE.Query Total': ('query', True),
    'QCSE.Semantic Routing': ('query', False), 'QCSE.Metadata Binding': ('frame', False),
    'QCSE.Direct Answer': ('query', True), 'QCSE.Candidate Retrieval': ('query', False),
    'QCSE.Candidate Construction': ('query', False),
    'QCSE.Verification and Continuation': ('query', True),
    'QCSE.Native Generation': ('query', True),
    'Frame E2E': ('frame', False), 'Query E2E': ('query', True),
    'Frame Forward': ('frame', False),
}
MODULES = ('OGSM.Total', 'DAVP.Total', 'QCSE.Total', 'Original Perception')


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def aggregate_samples(rows):
    """Sum only disjoint invocations of the SAME component in one frame/query."""
    grouped = defaultdict(list)
    for r in rows:
        if r['execution_status'] == 'COMPLETED':
            key = tuple(r.get(k, '') for k in ('trial_id', 'sequence_id', 'frame_id', 'query_id', 'component_name', 'execution_path', 'sample_unit'))
            grouped[key].append(r)
    result = []
    for items in grouped.values():
        row = dict(items[0], aggregation='frame_sum' if items[0]['sample_unit'] == 'frame' else 'query_sum',
                   invocation_count=sum(int(r['invocation_count']) for r in items))
        for metric in METRICS:
            values = [float(r[metric]) for r in items if r.get(metric) not in ('', None)]
            row[metric] = sum(values) if len(values) == len(items) else None
        result.append(row)
    # Same measured query scope, exported under its module name as well as E2E.
    # This is an inclusive alias, never a second executed stage.
    result += [dict(r, component_name='QCSE.Query Total', parent_component='QCSE.Total',
                    aggregation='alias_of_Query_E2E') for r in list(result) if r['component_name']=='Query E2E']
    # Two explicitly disjoint module sums, never added to their children later.
    for name, children in [('OGSM.Total', ('OGSM.Precheck', 'OGSM.Commit')),
                           ('Original Perception', ('Detection Head', 'Mapping Head'))]:
        by_frame = defaultdict(list)
        for r in result:
            if r['component_name'] in children:
                by_frame[(r['trial_id'], r['sequence_id'], r['frame_id'])].append(r)
        for items in by_frame.values():
            if {r['component_name'] for r in items} != set(children):
                raise ValueError('Incomplete module components: ' + name)
            row = dict(items[0], component_name=name, parent_component='', aggregation='disjoint_children_sum')
            for metric in METRICS:
                values = [float(r[metric]) for r in items if r.get(metric) not in ('', None)]
                row[metric] = sum(values) if len(values) == len(items) else None
            row['invocation_count'] = 1
            result.append(row)
    return result


def summarize_trial(rows, trial_id, frames, queries, frame_component='Frame E2E'):
    result = []
    for component, (unit, gpu) in COMPONENTS.items():
        if component in ('Frame E2E', 'Frame Forward') and component != frame_component:
            continue
        selected = [r for r in rows if r['component_name'] == component]
        # Export all meaningful clocks independently. Empty conditional paths stay empty.
        clocks = ('cpu_host_ms', 'gpu_device_ms') if gpu else ('cpu_host_ms',)
        if component in ('Frame E2E', 'Frame Forward', 'Query E2E'):
            clocks = ('component_wall_ms',)
        for metric in clocks:
            values = [float(r[metric]) for r in selected if r.get(metric) not in ('', None)]
            denominator = frames if unit == 'frame' else queries
            e2e_name = frame_component if unit == 'frame' else 'Query E2E'
            e2e = [float(r['component_wall_ms']) for r in rows if r['component_name'] == e2e_name]
            result.append(dict(trial_id=trial_id, component_name=component, metric=metric,
                sample_unit=unit, statistical_unit=unit, sample_count=len(values),
                invocation_count=sum(int(r['invocation_count']) for r in selected),
                mean_ms=float(np.mean(values)) if values else None,
                p95_ms=float(np.percentile(values, 95, method='linear')) if values else None,
                std_ms=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                invocation_rate=len(selected) / denominator if denominator else None,
                global_average_ms=sum(values) / denominator if denominator and values else (0. if not selected else None),
                e2e_share_percent=100 * sum(values) / sum(e2e) if values and sum(e2e) else None,
                share_definition='nonadditive host/enqueue or device-span ratio; not a critical-path partition'))
            result[-1]['share_denominator'] = e2e_name
    return result


def summarize_trials(trials):
    grouped = defaultdict(list)
    for r in trials:
        grouped[(r['component_name'], r['metric'], r['sample_unit'])].append(r)
    result = []
    def mean(values):
        return float(np.mean(values)) if values else None
    def std(values):
        return float(np.std(values, ddof=1)) if len(values) > 1 else None
    for (component, metric, unit), items in grouped.items():
        means = [float(r['mean_ms']) for r in items if r['mean_ms'] not in ('', None)]
        p95s = [float(r['p95_ms']) for r in items if r['p95_ms'] not in ('', None)]
        result.append(dict(component_name=component, metric=metric, sample_unit=unit,
            share_denominator=items[0].get('share_denominator', 'Frame E2E' if unit == 'frame' else 'Query E2E'),
            mean_ms=mean(means), mean_std_ms=std(means), p95_ms=mean(p95s), p95_std_ms=std(p95s),
            samples=sum(int(r['sample_count']) for r in items), trials_with_samples=len(means), trials_requested=len(items),
            min_samples_per_trial=min(int(r['sample_count']) for r in items),
            invocation_rate=mean([float(r['invocation_rate']) for r in items if r['invocation_rate'] not in ('', None)]),
            global_average_ms=mean([float(r['global_average_ms']) for r in items if r['global_average_ms'] not in ('', None)]),
            e2e_share_percent=mean([float(r['e2e_share_percent']) for r in items if r['e2e_share_percent'] not in ('', None)]),
            p95_caution='sparse_or_missing_trials' if any(int(r['sample_count']) < 20 for r in items) else ''))
    return result


def paths(receipts, trial_id):
    counts = defaultdict(int)
    frames, queries = len(receipts), 0
    for r in receipts:
        counts['global_residual_reuse' if r['visual_diagnostics']['approx_global_residual_reuse'] else 'global_refresh'] += 1
        for q in r['requests']:
            queries += 1
            counts['direct_answer'] += int(q.get('llm_skipped', False))
            counts['candidate_token_verification'] += int(q.get('verification_forward_calls', 0) > 0)
            counts['native_generation_without_verification'] += int(not q.get('llm_skipped') and not q.get('verification_forward_calls', 0))
            # A rejection can terminate at EOS. Count fallback only when actual
            # output contains a subsequent token, proving continuation occurred.
            counts['fallback_after_rejection'] += int(any(
                log.get('rejection_position') is not None and
                len(q.get('output_token_ids', [])) > log['output_offset'] + log['accepted'] + 1
                for log in q.get('verification_log', [])))
    return [dict(trial_id=trial_id, execution_path=p, invocation_count=counts[p],
                 sample_unit='frame' if p.startswith('global_') else 'query',
                 denominator=frames if p.startswith('global_') else queries,
                 invocation_rate=counts[p] / (frames if p.startswith('global_') else queries),
                 timing_scope=('merged_with_candidate_verification; isolated latency unavailable'
                               if p == 'fallback_after_rejection' else 'see component conditional latency'))
            for p in ('global_residual_reuse', 'global_refresh', 'direct_answer', 'candidate_token_verification', 'native_generation_without_verification', 'fallback_after_rejection')]


def validate_counts(raw, receipts, manifest):
    from collections import Counter
    frame_component = manifest.get('timing', {}).get('frame_component', 'Frame E2E')
    expected = [(s['scene_id'], t) for s in manifest['scenes'] for t in s['tokens']]
    if [(r['sequence_id'], r['frame_id']) for r in receipts] != expected:
        raise ValueError('Missing/reordered frames relative to prepared workload')
    by_frame = defaultdict(list)
    for row in raw:
        by_frame[(row['sequence_id'], row['frame_id'])].append(row)
    if set(by_frame) != set(expected):
        raise ValueError('Raw component records do not match input frames')
    for receipt in receipts:
        rows = by_frame[(receipt['sequence_id'], receipt['frame_id'])]
        counts = Counter(r['component_name'] for r in rows)
        for name in ('OGSM.Precheck','OGSM.Commit','DAVP.Total','DAVP.Patch Embedding',
                     'DAVP.Incremental EVA','Detection Head','Mapping Head','QCSE.Total',
                     'QCSE.Metadata Binding',frame_component):
            if counts[name] != 1:
                raise ValueError(f'Wrong calls for {name}: {counts[name]}')
        if counts['DAVP.Visual Planning'] != 2 or counts['DAVP.Global Refresh'] + counts['DAVP.Global Residual Reuse'] != 1:
            raise ValueError('Wrong DAVP stages or nonexclusive global paths')
        queries = len(receipt['questions'])
        if queries != 5 or len(receipt['requests']) != queries:
            raise ValueError('Wrong official factual query count')
        for name in ('Query E2E','QCSE.Semantic Routing'):
            if counts[name] != queries:
                raise ValueError('Missing query timing: '+name)
        for qi, request in enumerate(receipt['requests']):
            selected = Counter(r['component_name'] for r in rows if str(r.get('query_id')) == str(qi))
            direct = bool(request.get('llm_skipped'))
            verified = request.get('verification_forward_calls',0) > 0
            if (selected['QCSE.Direct Answer'] != int(direct) or
                selected['QCSE.Verification and Continuation'] != int(verified) or
                selected['QCSE.Native Generation'] != int(not direct and not verified) or
                selected['QCSE.Candidate Retrieval'] != int(not direct) or
                selected['QCSE.Candidate Construction'] != int(not direct)):
                raise ValueError('Query path timings disagree with real execution diagnostics')


def plot(out, rows, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r['component_name'] not in (*MODULES, 'QCSE.Query Total', 'Frame E2E', 'Frame Forward', 'Query E2E')
                and r['metric'] == ('gpu_device_ms' if COMPONENTS[r['component_name']][1] else 'cpu_host_ms')]
    directory = out / 'figures'
    directory.mkdir()
    def bars(items, value, error, name, title):
        fig, ax = plt.subplots(figsize=(max(10, len(items)*.65), 5))
        valid = [(i, r) for i, r in enumerate(items) if r[value] is not None]
        ax.bar([i for i, r in valid], [r[value] for i, r in valid],
               yerr=[r[error] if r[error] is not None else 0 for i, r in valid], capsize=3, color='#367a9f')
        for i, r in enumerate(items):
            if r[value] is None:
                ax.text(i, 0, 'N/A\nN=0', ha='center', fontsize=8)
        ax.set_xticks(range(len(items)), [r['component_name'] + ('\nGPU span' if r['metric']=='gpu_device_ms' else '\nHost') for r in items], rotation=55, ha='right', fontsize=8)
        ax.set_ylabel('Latency (ms)')
        ax.set_title(title + (' — SMOKE ONLY' if smoke else ''))
        ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        for suffix in ('pdf', 'png'):
            fig.savefig(directory / (name+'.'+suffix), dpi=320)
        plt.close(fig)
    bars(selected, 'mean_ms', 'mean_std_ms', 'mean_component_latency', 'Mean component latency (conditional paths; mixed clocks)')
    bars(selected, 'p95_ms', 'p95_std_ms', 'p95_component_latency', 'P95 component latency (conditional paths; mixed clocks)')
    modules = [r for r in rows if r['component_name'] in MODULES and r['metric'] == 'cpu_host_ms']
    # Common host clock for module comparison, no invalid additive stack.
    bars(modules, 'mean_ms', 'mean_std_ms', 'module_latency_breakdown', 'Module host latency — inclusive totals; bars are not additive')


def report(out):
    out = Path(out)
    manifest = read(out / 'experiment_config.json')
    frame_component = manifest.get('timing', {}).get('frame_component', 'Frame E2E')
    summaries, path_rows, fingerprints = [], [], []
    for directory in sorted(p for p in out.glob('trial_*') if p.is_dir()):
        status = read(directory / 'completion.json')
        if status['status'] != 'COMPLETED':
            raise RuntimeError('Incomplete trial: ' + str(directory))
        receipts = list(jsonl(directory / 'receipts.jsonl'))
        fingerprints.append([(r['sequence_id'], r['frame_id'], r['questions'], r['input_digest'], r['answers']) for r in receipts])
        raw = read_csv(directory / 'raw_component_latency.csv')
        if any(r['execution_status'] != 'COMPLETED' for r in raw):
            raise RuntimeError('Failed component invocation')
        validate_counts(raw, receipts, manifest)
        samples = aggregate_samples(raw)
        csv_write(directory / 'frame_component_latency.csv', samples)
        trial = summarize_trial(samples, directory.name, len(receipts), sum(len(r['questions']) for r in receipts), frame_component=frame_component)
        csv_write(directory / 'trial_component_summary.csv', trial)
        summaries += trial
        path_rows += paths(receipts, directory.name)
    if len(fingerprints) != manifest['effective_trials'] or any(f != fingerprints[0] for f in fingerprints):
        raise RuntimeError('Trial input/order/query/output mismatch or missing trials')
    result = summarize_trials(summaries)
    csv_write(out / 'trial_component_summary.csv', summaries)
    csv_write(out / 'component_latency_summary.csv', result)
    csv_write(out / 'module_latency_summary.csv', [r for r in result if r['component_name'] in MODULES])
    overall = [r for r in result if r['component_name'] in (frame_component, 'Query E2E')]
    csv_write(out / 'e2e_latency_summary.csv', overall)
    if frame_component == 'Frame Forward':
        csv_write(out / 'forward_latency_summary.csv', overall)
    csv_write(out / 'execution_path_summary.csv', path_rows)
    plot(out, result, manifest['smoke'])
    write(out / 'completion.json', dict(status='COMPLETED', trials=len(fingerprints), smoke=manifest['smoke'],
        identical_inputs_queries_answers=True, std_definition='sample SD across independent trial Mean/P95, ddof=1',
        frame_component=frame_component, timing_scope=manifest.get('timing', {}),
        latency_definition='component conditional invocation/frame sums; clocks separate; no pooled cross-trial SD'))
