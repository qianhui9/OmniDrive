"""Offline selected-query analysis; reconstructed totals are NOT three-query runs."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

KEEP = (0, 1, 3)
DROP = (2, 4)
CLOCKS = ('cpu_host_ms', 'gpu_device_ms', 'component_wall_ms')


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8*1024**2), b''):
            h.update(data)
    return h.hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(trial, label, metric, unit, values, interpretation):
    values = [float(v) for v in values]
    assert all(np.isfinite(v) and v >= 0 for v in values)
    return dict(trial_id=trial, component_name=label, metric=metric, sample_unit=unit,
        interpretation=interpretation, sample_count=len(values),
        mean_ms=float(np.mean(values)) if values else None,
        p95_ms=float(np.percentile(values, 95, method='linear')) if values else None)


def run(source):
    source = Path(source).resolve()
    manifest = json.loads((source/'experiment_config.json').read_text())
    completion = json.loads((source/'completion.json').read_text())
    assert completion['status'] == 'COMPLETED' and not completion['smoke']
    assert completion['frame_component'] == 'Frame Forward'
    out = source/'three_query_subset_q0_q1_q3'
    assert not out.exists(), 'Refuse to overwrite an existing offline report'
    frame_rows, trial_rows, path_rows, source_hashes = [], [], [], {}
    frame_identity = []
    for directory in sorted(p for p in source.glob('trial_*') if p.is_dir()):
        status = json.loads((directory/'completion.json').read_text())
        assert status['status'] == 'COMPLETED'
        for name, key in [('raw_component_latency.csv', 'raw_sha256'), ('receipts.jsonl', 'receipts_sha256')]:
            actual = sha(directory/name)
            assert actual == status[key]
            source_hashes[str((directory/name).relative_to(source))] = actual
        raw = read_csv(directory/'raw_component_latency.csv')
        assert all(r['execution_status']=='COMPLETED' for r in raw)
        by_frame = defaultdict(list)
        for row in raw:
            by_frame[(row['sequence_id'], row['frame_id'])].append(row)
        with (directory/'receipts.jsonl').open() as f:
            receipts = [json.loads(line) for line in f if line.strip()]
        assert len(receipts) == manifest['measured_frames']
        ids = []
        query_values = defaultdict(list)
        paths = defaultdict(int)
        current_frames = []
        for receipt in receipts:
            key = (receipt['sequence_id'], receipt['frame_id'])
            ids.append(key)
            assert receipt['groups'] == ['vehicle_category','vehicle_existence','pedestrian_position','other','other']
            assert len(receipt['questions']) == len(receipt['requests']) == 5
            rows = by_frame[key]
            def unique(name, query=''):
                selected = [r for r in rows if r['component_name']==name and r['query_id']==query]
                assert len(selected)==1, (key, name, query, len(selected))
                return selected[0]
            queries = {i:unique('Query E2E',str(i)) for i in range(5)}
            full = float(unique('Frame Forward')['component_wall_ms'])
            assert np.isclose(full,receipt['forward_ms'],rtol=0,atol=1e-9)
            removed_wall = sum(float(queries[i]['component_wall_ms']) for i in DROP)
            estimated = full-removed_wall
            record = dict(trial_id=directory.name, sequence_id=key[0], frame_id=key[1],
                retained_query_ids='0,1,3', removed_query_ids='2,4',
                original_five_query_forward_wall_ms=full,
                removed_query_wall_sum_ms=removed_wall,
                reconstructed_three_query_forward_wall_ms=estimated,
                measured_selected_query_wall_sum_ms=sum(float(queries[i]['component_wall_ms']) for i in KEEP))
            assert estimated > 0 and estimated <= full
            for metric in ('cpu_host_ms','gpu_device_ms'):
                total = float(unique('QCSE.Total')[metric])
                removed = sum(float(queries[i][metric]) for i in DROP)
                record['reconstructed_qcse_'+metric] = total-removed
                assert total-removed >= 0
            for label, children in [('OGSM.Total',('OGSM.Precheck','OGSM.Commit')),
                                    ('DAVP.Total',('DAVP.Total',)),
                                    ('Original Perception',('Detection Head','Mapping Head'))]:
                for metric in (['cpu_host_ms'] if label=='OGSM.Total' else ['cpu_host_ms','gpu_device_ms']):
                    record[label+'_'+metric] = sum(float(unique(child)[metric]) for child in children)
            for qi in KEEP:
                request=receipt['requests'][qi]
                paths['direct_answer'] += int(bool(request.get('llm_skipped')))
                paths['candidate_verification'] += int(request.get('verification_forward_calls',0)>0)
                paths['native_generation_without_verification'] += int(not request.get('llm_skipped') and not request.get('verification_forward_calls',0))
                paths['fallback_after_rejection'] += int(any(log.get('rejection_position') is not None and len(request.get('output_token_ids',[]))>log['output_offset']+log['accepted']+1 for log in request.get('verification_log',[])))
                for metric in CLOCKS:
                    query_values[('Selected Query',metric)].append(float(queries[qi][metric]))
                    query_values[('Original q'+str(qi)+' — '+receipt['groups'][qi],metric)].append(float(queries[qi][metric]))
                for row in rows:
                    if row['query_id']==str(qi) and row['component_name']!='Query E2E':
                        for metric in CLOCKS:
                            if row.get(metric) not in ('',None):
                                query_values[(row['component_name'],metric)].append(float(row[metric]))
            frame_rows.append(record)
            current_frames.append(record)
        frame_identity.append(ids)
        for label, metric, field, interpretation in [
            ('Three-query Frame Forward (reconstructed)','component_wall_ms','reconstructed_three_query_forward_wall_ms','offline_reconstruction_not_counterfactual_measurement'),
            ('Original five-query Frame Forward','component_wall_ms','original_five_query_forward_wall_ms','measured_five_query_workload'),
            ('QCSE three-query total (reconstructed)','cpu_host_ms','reconstructed_qcse_cpu_host_ms','offline_reconstruction_not_counterfactual_measurement'),
            ('QCSE three-query total (reconstructed)','gpu_device_ms','reconstructed_qcse_gpu_device_ms','offline_reconstruction_not_counterfactual_measurement'),
            ('Selected three-query sum','component_wall_ms','measured_selected_query_wall_sum_ms','measured_query_sum_excludes_batch_shared_cost')]:
            trial_rows.append(stats(directory.name,label,metric,'frame',[r[field] for r in current_frames],interpretation))
        for label in ('OGSM.Total','DAVP.Total','Original Perception'):
            for metric in (['cpu_host_ms'] if label=='OGSM.Total' else ['cpu_host_ms','gpu_device_ms']):
                trial_rows.append(stats(directory.name,label,metric,'frame',[r[label+'_'+metric] for r in current_frames],'measured_shared_stage_in_five_query_run'))
        for (label,metric),values in query_values.items():
            trial_rows.append(stats(directory.name,label,metric,'query',values,'measured_retained_queries_in_five_query_run'))
        for label,count in paths.items():
            path_rows.append(dict(trial_id=directory.name,execution_path=label,requests=count,
                denominator=len(receipts)*3,invocation_rate=count/(len(receipts)*3),
                interpretation='observed_subset_of_five_query_execution; fallback_is_verification_subset'))
    assert len(frame_identity)==manifest['effective_trials'] and all(x==frame_identity[0] for x in frame_identity)
    grouped=defaultdict(list)
    for row in trial_rows:
        grouped[(row['component_name'],row['metric'],row['sample_unit'],row['interpretation'])].append(row)
    summary=[]
    for (label,metric,unit,interpretation),items in grouped.items():
        summary.append(dict(component_name=label,metric=metric,sample_unit=unit,
            interpretation=interpretation,trials=len(items),samples=sum(x['sample_count'] for x in items),
            mean_ms=float(np.mean([x['mean_ms'] for x in items])),
            mean_std_ms=float(np.std([x['mean_ms'] for x in items],ddof=1)) if len(items)>1 else None,
            p95_ms=float(np.mean([x['p95_ms'] for x in items])),
            p95_std_ms=float(np.std([x['p95_ms'] for x in items],ddof=1)) if len(items)>1 else None))
    out.mkdir()
    write_csv(out/'frame_reconstruction.csv',frame_rows)
    write_csv(out/'trial_summary.csv',trial_rows)
    write_csv(out/'latency_summary.csv',summary)
    write_csv(out/'execution_paths.csv',path_rows)
    config=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),source=str(source),
        scope='offline_subset_and_reconstruction_only; no GPU rerun',retained_query_ids=KEEP,removed_query_ids=DROP,
        selection=['What type of vehicle is directly ahead?','Is there a car in front?','first original QA per frame'],
        independent_source_trials=manifest['effective_trials'],frames_per_trial=manifest['measured_frames'],
        frame_wall_formula='original Frame Forward wall - Query E2E wall(q2) - Query E2E wall(q4), calculated frame by frame',
        qcse_formula='original QCSE.Total - removed Query E2E using the SAME host/device clock separately',
        statistics='mean and linear P95 per source trial; average and sample SD(ddof=1) across source trials',
        assumptions='retained query outputs/costs and shared state/costs fixed at original five-query execution',
        limitations=['reindexed query RNG changes in a real three-query run','answer-memory and acceptance-state history may change','shared overhead and device thermal/scheduling effects retained from original workload','reported SD is reconstruction repeatability, not prediction error'],
        raw_source_sha256=source_hashes)
    (out/'analysis_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    return out,summary,trial_rows,path_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    out,summary,trials,paths=run(args.source)
    def pair(r,kind='mean'):return f"{r[kind+'_ms']:.4f} ± {r[kind+'_std_ms']:.4f}"
    lookup={(r['component_name'],r['metric']):r for r in summary}
    def table(headers,rows):
        return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]+['']
    lines=['# 每帧三问：子集实测与整帧离线重构','',
        '**本报告没有重新运行三问负载。整帧和 QCSE 批次总时延为离线重构估算，不能称为实际三问推理实测。** 单查询及共享阶段数值来自原五问运行的实测记录。','',
        '保留原 query_id **0、1、3**：前方车辆类型、前方是否有汽车、每帧第一个原始 QA；移除最近行人位置（q2）及第二个原始 QA（q4）。原顺序/随机种子和缓存历史保留在原五问执行环境中。','',
        '源数据为三轮独立正式测量，每轮 723 帧；三问子集每轮 2169 查询，累计 2169 帧次 / 6507 查询次。输入加载、外部预处理、H2D、外部问题编码及输出导出已经在源数据的 Frame Forward 外。','',
        '## 1. 时延结果','']
    names=[('Three-query Frame Forward (reconstructed)','component_wall_ms','三问整帧纯前向（估算）'),
           ('QCSE three-query total (reconstructed)','cpu_host_ms','QCSE 三问批次 Host（估算）'),
           ('QCSE three-query total (reconstructed)','gpu_device_ms','QCSE 三问批次 GPU span（估算）'),
           ('Selected three-query sum','component_wall_ms','三问查询区间之和（原运行实测，未含共享准备）'),
           ('Selected Query','component_wall_ms','保留问题的单查询（原运行实测）')]
    lines+=table(['指标','单位','Mean ± 跨轮 SD (ms)','P95 ± 跨轮 SD (ms)','N'],[[label,lookup[(name,metric)]['sample_unit'],pair(lookup[(name,metric)]),pair(lookup[(name,metric)],'p95'),lookup[(name,metric)]['samples']] for name,metric,label in names])
    lines+=['## 2. 帧级模块','',
        '共享阶段保留五问运行中的原实测值，未按 3/5 缩放；QCSE 为同一时钟逐帧扣除两问后的估算。模块含父子/包含区间，不相加求整帧 P95。','']
    names=[('OGSM.Total','cpu_host_ms','原共享阶段实测'),('DAVP.Total','cpu_host_ms','原共享阶段实测'),('DAVP.Total','gpu_device_ms','原共享阶段实测'),('Original Perception','cpu_host_ms','原共享阶段实测'),('Original Perception','gpu_device_ms','原共享阶段实测'),('QCSE three-query total (reconstructed)','cpu_host_ms','离线估算'),('QCSE three-query total (reconstructed)','gpu_device_ms','离线估算')]
    lines+=table(['模块','时钟','性质','Mean ± SD (ms/帧)','P95 ± SD (ms/帧)'],[[name,'Host' if metric=='cpu_host_ms' else 'GPU span',nature,pair(lookup[(name,metric)]),pair(lookup[(name,metric)],'p95')] for name,metric,nature in names])
    lines+=['## 3. 保留的各类问题','']
    questions=[r for r in summary if r['component_name'].startswith('Original q') and r['metric']=='component_wall_ms']
    lines+=table(['原问题位置与分组','Mean ± SD (ms/查询)','P95 ± SD (ms/查询)','N'],[[r['component_name'],pair(r),pair(r,'p95'),r['samples']] for r in questions])
    lines+=['## 4. 每轮重构结果','']
    selected=[r for r in trials if r['component_name']=='Three-query Frame Forward (reconstructed)']
    lines+=table(['源轮次','帧数','重构 Mean (ms/帧)','重构 P95 (ms/帧)'],[[r['trial_id'],r['sample_count'],f"{r['mean_ms']:.4f}",f"{r['p95_ms']:.4f}"] for r in selected])
    lines+=['## 5. 保留查询的原执行路径','',
        '下表仅描述原五问运行中所选三问的路径，不能保证真实三问运行仍有相同比例。Fallback 为验证集合的子集，不加入主路径分母。','']
    grouped=defaultdict(list)
    for row in paths:grouped[row['execution_path']].append(row)
    lines+=table(['路径','每轮请求数','三轮累计','占保留查询比例'],[[name,','.join(str(r['requests']) for r in rows),sum(r['requests'] for r in rows),f"{100*np.mean([r['invocation_rate'] for r in rows]):.3f}%"] for name,rows in grouped.items()])
    original=lookup[('Original five-query Frame Forward','component_wall_ms')]['mean_ms']
    reconstructed=lookup[('Three-query Frame Forward (reconstructed)','component_wall_ms')]['mean_ms']
    lines+=['## 6. 方法与解释限制','',
        '`三问整帧估算(f) = 原五问 Frame Forward wall(f) - q2 的 Query E2E wall(f) - q4 的 Query E2E wall(f)`。逐帧扣除后，分别在每个源轮次计算 Mean 与线性插值 P95，最后计算三轮均值与样本 SD（ddof=1）。没有将 P95 直接相减或将整帧乘以 3/5。','',
        'QCSE 批次分别使用 Host 减 Host、GPU span 减 GPU span；不将两种时钟混算。整帧保留原共享阶段与未单列开销。GPU span 不是纯 kernel busy time。','',
        f'重构 Mean 相对原五问实测 {original:.4f} ms 减少 {original-reconstructed:.4f} ms（{100*(1-reconstructed/original):.3f}%）。这只是按所选问题扣除的算术结果，不是实际三问运行的已验证加速或独立实验结论。','',
        '实际三问运行会将原 q3 重新编号为 q2，而采样种子依赖问题序号；被删除查询也不再更新跨帧答案记忆和候选接受统计。共享开销、频率和调度也可能变化。因此真实三问结果、候选路径及生成长度可能不同；上述跨轮 SD 只表示基于原数据重构的波动，不涵盖反事实估算误差。','',
        '若需论文主表的真实三问纯前向测量，应使用固定三问协议重新运行；本报告可以作为明确标注的离线负载估算。','',
        '## 7. 数据与复现','',
        '[逐帧重构 CSV](frame_reconstruction.csv)、[逐轮统计](trial_summary.csv)、[跨轮汇总](latency_summary.csv)、[查询路径](execution_paths.csv)、[选择规则/假设/源文件哈希](analysis_config.json)、[原五问报告](../结果.md)。原正式实验的 CSV/收据保持原样。','',
        '复现命令（输出目录已存在时拒绝覆盖）：','',
        '```bash',
        'taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m tools.component_latency_eval.three_query_subset \\\n  --source tools/component_latency_eval/outputs/20261117_191525_562820',
        '```','']
    (out/'结果.md').write_text('\n'.join(lines))
    print('OUTPUT',out)
    for name,metric,label in names:
        print(label,name,metric,pair(lookup[(name,metric)]),pair(lookup[(name,metric)],'p95'))
    print('FRAME_RECONSTRUCTED',pair(lookup[('Three-query Frame Forward (reconstructed)','component_wall_ms')]),pair(lookup[('Three-query Frame Forward (reconstructed)','component_wall_ms')],'p95'))


if __name__=='__main__':
    main()
