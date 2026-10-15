"""Cross-artifact checks and supplementary metrics; all values derived from saved runs."""
import json
from pathlib import Path
import numpy as np
from .io_utils import write_json


def read(p):return json.loads(Path(p).read_text())


def finalize_artifacts(root):
    from worldstream.m1.mask_codec import decode_mask
    verification=[];example=None
    for path in sorted((root/'optimized_replay/scene_results').glob('*/new_full/state.jsonl')):
        original=root/'scene_results'/path.parent.parent.name/'new_full'
        prior=[json.loads(line)['precheck'] for line in (original/'state.jsonl').read_text().splitlines()]
        current=[json.loads(line)['precheck'] for line in path.read_text().splitlines()]
        assert len(current)==len(prior)
        for index,(a,b) in enumerate(zip(prior,current)):
            assert a['valid_anchor_ids']==b['valid_anchor_ids'] and a['invalid_anchor_ids']==b['invalid_anchor_ids']
            ma=np.load(original/f'masks/{index:03d}.npz');mb=np.load(path.parent/f'masks/{index:03d}.npz')
            for cid,v in b['invalidation_regions'].items():
                mask=decode_mask(v['exact_invalidation_mask'])
                assert np.array_equal(mask,ma[cid+'_invalid']) and np.array_equal(mask,mb[cid+'_invalid'])
                assert not v['transformer_dependency_certified']
            if b['valid_anchor_ids'] and example is None:example=b
        verification.append(dict(scene_run=path.parent.parent.name,frames=len(current),all_candidates_and_exact_masks_equal=True))
    write_json(root/'optimization_full_replay_checks.json',verification)
    diagnosis=root/'diagnosis.md'
    if diagnosis.exists() and 'diagnosis_independent.md' not in diagnosis.read_text():
        with diagnosis.open('a') as handle:
            handle.write('\n补充：为解除旧短路日志对遮挡/外观条件的遮蔽，完成了[逐约束独立测量](diagnosis_independent.md)。首场景加入独立遮挡检查后327→117，再由缺失可见性证据117→0；未可测条件以null保留。\n')
    if example:write_json(root/'m2_example.json',example)
    import cv2
    sample_ids=read(root/'protocol.json')['scene_splits']['scene_00']['samples']
    for record in read(root/'visualizations/mask_counts.json'):
        index=sample_ids.index(record['frame_id']);cid=record['camera']
        exact=np.load(root/f'scene_results/scene_00/new_full/masks/{index:03d}.npz')[cid+'_invalid']
        png=cv2.imread(str(root/f'visualizations/{index:03d}_{cid}_invalidation.png'),cv2.IMREAD_GRAYSCALE)
        assert png is not None and np.array_equal(png>0,exact)
        assert int(exact.sum())==record['invalid_pixels']
    write_json(root/'visualizations/verification.json',dict(binary_png_equals_exact_npz=True,
        images=len(read(root/'visualizations/mask_counts.json'))))
    # Feature comparison coverage, measured on the same newly computed GPU sequence.
    feature=read(root/'feature_oracle_metrics.json')
    if feature.get('status')=='measured':
        rows=read(root/'gpu/feature_oracle_records.json')
        states=[json.loads(line)['precheck'] for line in (root/'gpu/feature_oracle_00.jsonl').read_text().splitlines()]
        all_keys={(r['frame_id'],aid) for r in states for aid in r['diagnostics']['anchors']}
        accepted_keys={(r['frame_id'],aid) for r in states for aid in r['valid_anchor_ids']}
        comparable={(r['frame_id'],r['anchor_id']) for r in rows if r['anchor_id'] is not None}
        feature['comparison_coverage']=dict(all_historical_anchor_trials=len(all_keys),comparable_anchor_trials=len(comparable),
            accepted_anchor_trials=len(accepted_keys),comparable_accepted_trials=len(comparable&accepted_keys),
            overall_fraction=len(comparable)/len(all_keys) if all_keys else None,
            accepted_fraction=len(comparable&accepted_keys)/len(accepted_keys) if accepted_keys else None,
            unknown_not_compared=len(all_keys-comparable))
        curve=[]
        for threshold in read(root/'protocol.json')['risk_thresholds']:
            accepted=set()
            for r in states:
                for aid,a in r['diagnostics']['anchors'].items():
                    views=[v for v in a['cameras'].values() if v['geometry_valid']]
                    if a['within_horizon'] and a['motion_known'] and views and all(v['visibility_and_coverage_valid']
                        and v['residual'] is not None and v['residual']<=threshold for v in views):accepted.add((r['frame_id'],aid))
            for layer in feature['layers']:
                xs=[r['relative_l2'] for r in rows if r['layer']==layer and r['anchor_id'] is not None and (r['frame_id'],r['anchor_id']) in accepted]
                curve.append(dict(layer=layer,threshold=threshold,valid_candidate_rate=len(accepted)/len(all_keys),
                    accepted_comparable_records=len(xs),mean_relative_l2=float(np.mean(xs)) if xs else None))
        feature['feature_coverage_curve']=curve;write_json(root/'feature_oracle_metrics.json',feature)
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4))
        for layer in feature['layers']:
            rs=[r for r in curve if r['layer']==layer and r['mean_relative_l2'] is not None]
            ax.plot([r['valid_candidate_rate']*100 for r in rs],[r['mean_relative_l2'] for r in rs],'-o',label=layer)
        ax.set_xlabel('Valid candidate rate (%)');ax.set_ylabel('Accepted aligned feature relative L2');ax.legend(fontsize=8)
        ax.set_title('Exploratory feature–coverage, one GPU scene')
        for ext in ('png','pdf'):fig.savefig(root/'figures'/('feature_coverage.'+ext),dpi=220,bbox_inches='tight')
        plt.close(fig)
    latency=read(root/'latency_profile.json');gpu={}
    for name in ('gpu','gpu_optimized'):
        if not (root/name/'results.json').exists():continue
        result=read(root/name/'results.json');modes={}
        for mode in ('baseline','old_strict','new_full'):
            runs=[r for r in result['runs'] if r['mode']==mode and not r['warmup']]
            samples=[t for r in runs for t in r['m1_timings']]
            mapping=dict(projection='projection',temporal_correspondence='temporal_correspondence',
                visibility_verification='visibility_verification',geometry_support='geometry_support',
                observation_validation='observation_validation',novelty_detection='unexplained_regions',
                state_update='state_update_ms',association='association_ms',input_adaptation='input_adapter_including_transfers',
                output_adaptation='output_adapter_including_transfers_ms',m1_overall_wall='m1_total_including_adapters_ms')
            stats={k:dict(p50=float(np.percentile([t[v] for t in samples],50)),p95=float(np.percentile([t[v] for t in samples],95))) for k,v in mapping.items()} if samples else {}
            if samples:
                values=[t['input_gpu_to_cpu_transfer']+t['output_gpu_to_cpu_transfer_ms'] for t in samples]
                stats['gpu_to_cpu_transfer']=dict(p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)))
            modes[mode]=dict(percentiles_ms=stats,frame_samples=sum(len(r['frame_wall_ms']) for r in runs),
                peak_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in runs),peak_reserved_bytes=max(r['gpu_peak_reserved_bytes'] for r in runs),
                frame_p50_p95_ms=result['summary'][mode]['frame_p50_p95_ms'])
        gpu[name]=dict(modes=modes,metadata=result['metadata'],observe_only_equal=result['observe_only_equal'])
    if len(gpu)==2:
        assert read(root/'gpu/baseline_digests.json')==read(root/'gpu_optimized/baseline_digests.json')
        latency['cross_run_baseline_digests_equal']=True
    latency['online_full_model']=gpu
    write_json(root/'latency_profile.json',latency)
    supplement=['','## 补充核对与可复现材料','',
        f"最终版本在全部 {sum(x['frames'] for x in verification)} 帧回放上，候选ID、NPZ精确mask、M1Result压缩mask解码全部一致。检查见optimization_full_replay_checks.json，接口实例见m2_example.json。",
        '更细的重叠约束、未可测条件和独立遮挡诊断见[diagnosis_independent.md](diagnosis_independent.md)。它在离线测量中打破短路以观测其余条件；实际在线首失败仍使用diagnosis.md原始日志口径。']
    if feature.get('comparison_coverage'): supplement+=['','Feature Oracle可比较覆盖：`'+json.dumps(feature['comparison_coverage'])+'`。可比较子集选择会引入偏差；未可比状态没有被赋予零误差。feature_coverage图仅描述一个GPU场景的预设阈值运行点，没有用于校准。']
    if 'gpu_optimized' in gpu:
        supplement+=['','最终GPU版本的组件P50/P95（毫秒，输入/输出适配包含其传输子项，不相加）：','',
            '| 部分 | 旧严格M1 | 新M1 |','|---|---:|---:|']
        old=gpu['gpu_optimized']['modes']['old_strict']['percentiles_ms'];new=gpu['gpu_optimized']['modes']['new_full']['percentiles_ms']
        for key in new:supplement.append(f"| {key} | {old[key]['p50']:.2f}/{old[key]['p95']:.2f} | {new[key]['p50']:.2f}/{new[key]['p95']:.2f} |")
        supplement+=['','| 模式 | 优化前在线M1 P50/P95 ms | 最终在线M1 P50/P95 ms | 最终峰值allocated GiB |','|---|---:|---:|---:|']
        for mode in ('old_strict','new_full'):
            a=gpu['gpu']['modes'][mode]['percentiles_ms']['m1_overall_wall'];b=gpu['gpu_optimized']['modes'][mode]
            c=b['percentiles_ms']['m1_overall_wall']
            supplement.append(f"| {mode} | {a['p50']:.2f}/{a['p95']:.2f} | {c['p50']:.2f}/{c['p95']:.2f} | {b['peak_allocated_bytes']/2**30:.3f} |")
    # Keep the five requested research answers as the end of the report.
    report=root/'refinement_report.md';body=report.read_text();position=body.index('## M2边界、未解决问题和结论')
    report.write_text(body[:position]+'\n'.join(supplement)+'\n\n'+body[position:])
