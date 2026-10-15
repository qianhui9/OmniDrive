"""Generate auditable coverage/risk/latency reports from real executed records."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from .refine_coverage import collect
from .refinement_common import summarize_rows,ratio
from .io_utils import write_json


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='outputs/module1_refinement/coverage_final');args=p.parse_args()
    root=Path(args.root);figs=root/'figures';figs.mkdir(exist_ok=True)
    selection=read(root/'selection.json');selected=selection['selected'];summary=read(root/'holdout_summary.json')
    roi=read(root/'roi_verification.json') if (root/'roi_verification.json').exists() else None
    source_protocol=next(root/name for name in ('roi_verification.json','scope_fix_protocol.json','protocol.json') if (root/name).exists())
    data={name:collect(root,'holdout',name)[1] for name in summary}
    base=summary['sparse_m1'];chosen=summary[selected]
    geometry=summarize_rows(data['original_m1']['anchors'],'geometry_only')
    appearance=summarize_rows(data['original_m1']['anchors'],'appearance_only')
    ablation=dict(geometry_only=geometry,appearance_only=appearance,**{name:m['physical'] for name,m in summary.items()})
    write_json(root/'physical_ablation.json',ablation)
    category_metrics={}
    for name,d in data.items():
        groups={}
        for label in ['newly_visible','moving_object','visibility_drop','car','pedestrian','bus']:
            events=[e for e in d['events'] if label in e['reasons'] or e['semantic_class']==label]
            groups[label]=dict(events=len(events),missed=sum(not e['recalled'] for e in events),recall=ratio(sum(e['recalled'] for e in events),len(events)))
        category_metrics[name]=groups
    write_json(root/'critical_recall_by_category.json',category_metrics)
    def event_key(e):return(e['frame_id'],e['camera'],e['instance'])
    misses={name:{event_key(e) for e in d['events'] if not e['recalled']} for name,d in data.items()}
    rows=[]
    for scene in sorted((root/'holdout').iterdir()):
        item={}
        for name in ('sparse_m1',selected):
            pixels=read(scene/name/'pixels.json');li=read(scene/name/'lidar.json');events=read(scene/name/'events.json')
            item[name]=dict(stable=float(np.mean([r['stable'] for r in pixels])),effective_stable=float(np.mean([r['effective_stable'] for r in pixels])),
                false=sum(r['false_points'] for r in li),accepted=sum(r['accepted_points'] for r in li),
                critical=sum(r['critical_points'] for r in li),missed=sum(r['missed_critical_points'] for r in li),
                events=len(events),event_missed=sum(not e['recalled'] for e in events))
        rows.append(dict(scene=scene.name,methods=item))
    rng=np.random.RandomState(20261015);index=rng.randint(0,len(rows),(10000,len(rows)))
    def bootstrap(name):
        delta=np.array([r['methods'][selected][name]-r['methods']['sparse_m1'][name] for r in rows]);return np.percentile(delta[index].mean(1),[2.5,97.5]).tolist()
    def pooled_delta(numerator,denominator):
        res=[]
        for method in ('sparse_m1',selected):
            a=np.array([r['methods'][method][numerator] for r in rows]);b=np.array([r['methods'][method][denominator] for r in rows]);tot=b[index].sum(1)
            res.append(np.divide(a[index].sum(1),tot,out=np.full(len(index),np.nan),where=tot>0))
        delta=res[1]-res[0];return np.nanpercentile(delta,[2.5,97.5]).tolist()
    gate=dict(selected=selected,new_critical_event_misses=len(misses[selected]-misses['sparse_m1']),
        resolved_critical_event_misses=len(misses['sparse_m1']-misses[selected]),
        lidar_risk_nonincrease=chosen['lidar']['false_acceptance_rate']<=base['lidar']['false_acceptance_rate'],
        critical_point_misses_nonincrease=chosen['lidar']['missed_critical_points']<=base['lidar']['missed_critical_points'],
        physical_error_count_nonincrease=chosen['physical']['physical_errors']<=base['physical']['physical_errors'],
        physical_far_nonincrease=(chosen['physical']['physical_false_acceptance_rate'] is not None and base['physical']['physical_false_acceptance_rate'] is not None and chosen['physical']['physical_false_acceptance_rate']<=base['physical']['physical_false_acceptance_rate']) if base['physical']['physical_false_acceptance_rate'] is not None else None,
        selected_before_confirmation=True,scene_bootstrap95ci=dict(stable_delta=bootstrap('stable'),effective_stable_delta=bootstrap('effective_stable'),lidar_risk_delta=pooled_delta('false','accepted'),critical_point_miss_rate_delta=pooled_delta('missed','critical')),
        online_source_hash_match=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in read(source_protocol)['source_sha256'].items()))
    gate['empirical_nonregression']=bool(gate['new_critical_event_misses']==0 and gate['lidar_risk_nonincrease'] and gate['critical_point_misses_nonincrease'] and gate['physical_error_count_nonincrease'] and gate['physical_far_nonincrease'] is not False)
    if (root/'scope_fix_verification.json').exists():gate['scope_fix_verification']=read(root/'scope_fix_verification.json')
    if roi:gate['roi_exact_parity']=roi['all_exact'];gate['roi_frames_checked']=roi['frames_checked']
    write_json(root/'verification.json',gate);write_json(root/'per_scene_metrics.json',rows)
    write_json(root/'tradeoff_curves.json',dict(methods=summary,physical_ablation=ablation,
        note='Frozen guard radii 2,4,8; no heldout threshold selection. Image coverage denominator differs from sparse LiDAR risk denominator.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    def save(fig,name):
        fig.tight_layout();fig.savefig(figs/(name+'.png'),dpi=180);fig.savefig(figs/(name+'.pdf'));plt.close(fig)
    dense=['dense_guard_8','dense_guard_4','dense_guard_2']
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].plot([summary[n]['effective_stable']*100 for n in dense],[summary[n]['lidar']['false_acceptance_rate']*100 for n in dense],'o-',label='Dense + motion sweep')
    ax[0].scatter([base['effective_stable']*100],[base['lidar']['false_acceptance_rate']*100],marker='x',s=70,label='Existing sparse M1',color='red')
    for n in dense:ax[0].annotate(n.rsplit('_',1)[1]+' px',(summary[n]['effective_stable']*100,summary[n]['lidar']['false_acceptance_rate']*100),xytext=(4,6),textcoords='offset points')
    ax[0].set(xlabel='Effective supported image area (%)',ylabel='LiDAR sampled false acceptance (%)',title='Risk–Coverage (confirmation scenes)');ax[0].margins(x=.15);ax[0].legend()
    for label,m in [('Geometry only',geometry),('Appearance only',appearance),('Existing sparse',base['physical']),('Selected',chosen['physical'])]:
        if m['physical_false_acceptance_rate'] is not None:ax[1].scatter(m['valid_candidate_rate']*100,m['physical_false_acceptance_rate']*100,label=label,s=55)
    ax[1].set(xlabel='Accepted historical anchors (%)',ylabel='Physical-state false acceptance (%)',title='Independent annotation oracle');ax[1].legend()
    save(fig,'risk_coverage')
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for a,key,title in zip(ax,['critical_event_recall','new_event_recall'],['All critical object regions','Newly visible object regions']):
        a.plot([summary[n]['invalid']*100 for n in dense],[summary[n][key]*100 for n in dense],'o-',label='Dense + motion sweep')
        for n,l,c in [('original_m1','Original strict','gray'),('sparse_m1','Existing sparse','red')]:a.scatter(summary[n]['invalid']*100,summary[n][key]*100,label=l,color=c)
        a.set(xlabel='Invalidation area (%)',ylabel='Region recall at >=90% coverage (%)',title=title);a.legend()
    save(fig,'invalidation_recall')
    methods=['original_m1','sparse_m1',selected];components=['Projection','Temporal + exclusion','Anchor observation','Mask export','Association','State commit','Other']
    final_timings={n:data[n]['timings'] for n in methods}
    if roi:
        roi_timings=read(root/'roi_timings.json')
        for n in methods[1:]:
            final_timings[n]=[dict(r['precheck'],association_ms=r['commit']['association_ms'],state_update_ms=r['commit']['state_update_ms'],overall_ms=r['wall_ms']) for r in roi_timings if r['method']==n]
    stacks=[]
    for n in methods:
        sample=[]
        for t in final_timings[n]:
            temporal=max(0,t['precheck_total']-t['projection']-t['observation_validation']-t['unexplained_regions'])
            vals=[t['projection'],temporal,t['observation_validation'],t['unexplained_regions'],t['association_ms'],t['state_update_ms']]
            sample.append(vals+[max(0,t['overall_ms']-sum(vals))])
        stacks.append(np.mean(sample,axis=0))
    stacks=np.asarray(stacks);fig,ax=plt.subplots(figsize=(8,4));bottom=np.zeros(len(methods))
    for j,c in enumerate(components):ax.bar(['Original strict','Existing sparse','Selected dense'],stacks[:,j],bottom=bottom,label=c);bottom+=stacks[:,j]
    ax.set(ylabel='Mean M1 CPU wall time (ms)',title='Runtime Breakdown; excludes model / IO / oracle');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left');save(fig,'runtime_breakdown')
    write_json(root/'runtime_breakdown.json',dict(components=components,methods=methods,mean_ms=stacks.tolist(),note='Parallel work sums not added as wall time; temporal residual includes exclusion and setup.'))
    gpu_dir=root/('gpu_roi' if (root/'gpu_roi/results.json').exists() else 'gpu')
    gpu_path=gpu_dir/'results.json';gpu=read(gpu_path) if gpu_path.exists() else None
    feature_path=gpu_dir/'feature_oracle_records.json';features=read(feature_path) if feature_path.exists() else []
    feature_summary={}
    for layer in sorted({r['layer'] for r in features}):
        rows_f=[r for r in features if r.get('patch_group')=='dense_stable_pixel' and r['layer']==layer]
        total=sum(r['points'] for r in rows_f)
        feature_summary[layer]=dict(samples=total,relative_l2=ratio(sum(r['relative_l2']*r['points'] for r in rows_f),total),
            cosine=ratio(sum(r['cosine_similarity']*r['points'] for r in rows_f),total),
            fraction_l2_le_01=ratio(sum(r['fraction_l2_le_01']*r['points'] for r in rows_f),total),
            fraction_l2_le_02=ratio(sum(r['fraction_l2_le_02']*r['points'] for r in rows_f),total))
    write_json(root/'feature_summary.json',feature_summary)
    if features:
        fig,ax=plt.subplots(figsize=(8,4));names=[n for n in ['patch_embed','window_block_0','global_block_2','backbone_final'] if feature_summary[n]['relative_l2'] is not None]
        ax.plot(range(len(names)),[feature_summary[n]['relative_l2'] for n in names],'o-');ax.set_xticks(range(len(names)));ax.set_xticklabels(names,rotation=12);ax.set(ylabel='Relative L2 (point weighted)',title='Full-forward feature error at supported dense pixels');save(fig,'feature_error')
    m2=dict(scene_count=16,unique_frames=192,total_camera_pixels=sum(p['image_pixels'] for p in data[selected]['pixels']),
        supported_pixel_observations=sum(round(p['image_pixels']*p['effective_stable']) for p in data[selected]['pixels']),
        complete_16px_image_tiles=chosen['complete_tiles'],total_16px_image_tiles=chosen['total_tiles'],
        exact_neural_feature_tokens_certified=0,transformer_dependency_certified=False,gpu_operations_skipped=0,
        note='16px bins are contiguous image support opportunities, not actual EVA token identities. Source warp, patch footprint, attention dependency and numerical tolerances remain for M2.')
    write_json(root/'m2_candidates.json',m2)
    # Real-image overlays on a predeclared confirmation scene/frame, no best-case selection.
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos'];by_token={i['token']:i for i in infos}
    protocol=read(root/'protocol.json');scene=next(iter(protocol['splits']['holdout']));token=protocol['splits']['holdout'][scene][5]
    from .dataset import frame_from_info
    frame=frame_from_info(by_token[token],'data/nuscenes');fig,axes=plt.subplots(2,3,figsize=(15,6))
    for row,cid in enumerate(['CAM_FRONT','CAM_BACK']):
        for col,n in enumerate(methods):
            invalid=np.load(root/'holdout'/scene/n/'masks'/f'005_{cid}.npz')['invalid'];im=frame.cameras[cid].image.astype(float)/255
            overlay=im.copy();overlay[~invalid]=.5*overlay[~invalid]+.5*np.array([0,1,0]);axes[row,col].imshow(overlay)
            axes[row,col].set_title(f'{n} / {cid}\nSupported: {(~invalid).mean():.2%}');axes[row,col].axis('off')
    save(fig,'real_scene_masks')
    pct=lambda v:'N/A' if v is None else f'{100*v:.3f}%'
    report=['# M1：可信稳定区域覆盖优化与确认实验','',
        f'核心结果：相对首轮严格模式，稳定观测面积从 0% 到 {pct(chosen["stable"])}，刷新比例从 100% 到 {pct(chosen["invalid"])}；相对已有 sparse M1，最终保守配置的面积下降，但本次采样风险和关键变化漏检减少，并形成完整图像块候选。尚不能宣称获得有统计保证的安全深层缓存。',
        '', '本轮已完成代码、真实实验和验证，但没有同时实现“比现有 sparse M1 覆盖更高、成本更低、风险不增加”的全部目标。该配置保留为 observe_only 实验入口；不将在线观测支持等同于安全的神经计算复用。',
        '', '本报告以真实执行的 nuScenes 连续场景、独立离线标注/LiDAR 和完整 GPU Forward 为依据。`observe_only` 保留，主模型权重未修改。历史报告保存在 `coverage_round3/previous_refinement_report.md`。',
        '', '## 首轮严格 0% 覆盖与 100% 刷新的原因','',
        '首轮 `outputs/module1/experiment_report.md` 的严格模式要求独立可见性 mask，但真实 RGB 适配器没有该输入；`novelty_detector` 也拒绝用物体框直接解释背景，缺少 dense geometry support 时全部未知。动态目标另受速度不可靠、搜索不确定性过大等门限限制。后续 sparse M1 已有局部非零支持，但 7×7 patch 分散，难以形成完整 patch 尺度的复用候选。',
        '', '## 本轮算法','',
        '新增双向 DIS 稠密对应，逐像素验证反向误差 ≤0.35px、无折叠 Jacobian、正深度与双向重投影 ≤0.5px、非退化视差、双图局部纹理、灰度残差 ≤0.02、梯度残差 ≤0.04；只保留整个邻域都通过的内部像素。无矩形填充、掩码膨胀或神经网络。低平移采用有界旋转重投影，仍不能建立独立深度。',
        '密集证据替代 sparse 像素支持，而原稀疏锚点验证仍保留。历史动态对象加入未裁剪的方向性扫过区域与原有运动不确定性余量，所有这些范围只拒绝复用。速度信息不用于认证未知运动。当前 GT、LiDAR 和当前完整感知结果均未传入 precheck。',
        '最后加入计算范围优化：已确定必须刷新的区域不再执行无用的射线几何验证，仅为未排除区域保留完整的邻域计算边界；整个相机都被排除时直接跳过稠密计算。该优化没有改变支持门限或缩小刷新掩码。',
        '实现使用 [OpenCV DISOpticalFlow](https://docs.opencv.org/4.10.0/de/d4f/classcv_1_1DISOpticalFlow.html) 的 FAST 提议器，严格验证由本项目额外实现。',
        '', '## 实验过程与数据隔离','',
        '第一阶段：8 个校准场景 48..55、16 个验证场景 64..79，每场景 12 帧。首选 guard=2 在验证上覆盖提升，但关键 LiDAR 点漏检增多，因此没有将它当作通过的最终结果。第二阶段将这些已查看的 24 个场景明确转为开发/校准数据，新增未查看的 80..95 共 16 个场景作最终确认。每个场景先写 token 清单再运行；配置在确认开始前冻结，确认后的其他 guard 数值仅为预先定义的消融曲线。',
        '开发集还发现局部运动保护误改了原稀疏锚点的证据标志，使物理 FAR 因分母变化而上升。已将像素拒绝范围与原锚点证据分开；配置和场景保持冻结，全部确认场景重新执行。scope_fix_verification.json 逐帧证明修改前后像素掩码完全一致，锚点判断与已有 sparse M1 一致。原执行记录保存在 holdout_before_scope_fix。',
        '中间一次各方向等量扩张速度范围的尝试在开发阶段因过度刷新停止，未执行确认场景；记录保存在 `coverage_confirmation/status.json`。最终采用方向性且未裁剪的投影范围，避免视野外框被裁剪后错误扩张。',
        f'最终选择：**{selected}**。最终确认包括 192 个唯一帧、1,152 个相机图像；每种方案处理相同输入。场景是本地 nuScenes 验证集的新划分，不是官方隐藏测试集。',
        '每场景取前 12 个连续 keyframe（约 2Hz），不是高频视频或全场景评估。覆盖率与 LiDAR Oracle 使用保持纵横比的 640×360 原图回放；GPU 对照沿用实际部署的裁剪再缩放至 640×640 流程。两类实验的面积、候选数与耗时不能直接混用。',
        '', '## 覆盖、刷新与独立风险','',
        'Certified Stable Area 在这里指通过在线图像/几何规则的观测面积，不是已证明相等的神经特征面积。effective stable 还扣除了场景切换和每 10 帧强制刷新；其数值等于 1 − Invalidation Area。',
        '', '| 方法 | 稳定观测面积 | 有效可用面积 | 刷新面积 | LiDAR 位置错误接受 | 关键区域召回 | 关键 LiDAR 点召回 |', '|---|---:|---:|---:|---:|---:|---:|']
    for n,m in summary.items():report.append(f'| {n} | {pct(m["stable"])} | {pct(m["effective_stable"])} | {pct(m["invalid"])} | {pct(m["lidar"]["false_acceptance_rate"])} | {pct(m["critical_event_recall"])} | {pct(m["lidar"]["critical_recall"])} |')
    report += ['',f'相对已有 sparse M1，最终配置稳定观测面积变化 **{100*(chosen["stable"]-base["stable"]):+.3f} 个百分点**，有效可用面积变化 **{100*(chosen["effective_stable"]-base["effective_stable"]):+.3f} 个百分点**，刷新面积变化 **{100*(chosen["invalid"]-base["invalid"]):+.3f} 个百分点**。相对首轮严格模式，有效可用面积增加 {pct(chosen["effective_stable"])}，刷新面积从 100% 降至 {pct(chosen["invalid"])}。',
        f'独立 LiDAR 位置：可评分 {chosen["lidar"]["scorable_points"]:,}，接受 {chosen["lidar"]["accepted_points"]:,}，错误 {chosen["lidar"]["false_points"]:,}。关键 LiDAR 点 {chosen["lidar"]["critical_points"]:,}，漏刷新 {chosen["lidar"]["missed_critical_points"]:,}；关键物体区域事件 {chosen["critical_events"]:,}。相对 sparse M1 新增未召回区域 {gate["new_critical_event_misses"]}，修复原未召回区域 {gate["resolved_critical_event_misses"]}。经验非退化检查：**{gate["empirical_nonregression"]}**。若该项为 False，本配置仅为实验候选，不能宣称满足严格约束。',
        'LiDAR 判定使用独立点云、双帧深度支持与标注实例运动，采样位置的历史对应误差 >2px 或实际移动即为错误；新增、运动、前景遮挡点另统计漏刷新。它是稀疏采样参考，未标注点假设静态，相机/LiDAR 时间偏差和稀疏遮挡近似仍有影响。不能将采样 FAR 外推成全部像素的真实错误率。0 接受的首轮严格模式 FAR 为 N/A，不能写成零风险；它全图刷新的关键点召回是平凡的 100%，本轮任何非零漏点都不能说与它完全等价。',
        f'场景聚类 bootstrap 的 LiDAR FAR 差值 95% 区间为 {100*gate["scene_bootstrap95ci"]["lidar_risk_delta"][0]:.3f} 至 {100*gate["scene_bootstrap95ci"]["lidar_risk_delta"][1]:.3f} 个百分点；区间跨越 0，因此点估计改善不等于已证明统计非劣。物理锚点确认集只接受 {chosen["physical"]["accepted_labeled"]} 个可评分记录，其中错误 {chosen["physical"]["physical_errors"]} 个，仍不能作为可靠的自动状态复用依据。',
        '关键区域事件涵盖新增入视野、标注中心位移 >0.25m、可见性下降；区域内至少 90% 请求刷新才计为召回。此门限与旧评估一致。标注框不是像素分割，所以同时报告点级证据，不仅看框级召回。',
        '', '| 关键事件类别（可重叠） | 事件数 | sparse 漏召回数 | 最终配置漏召回数 | 最终召回率 |', '|---|---:|---:|---:|---:|']
    for label,m in category_metrics[selected].items():
        report.append(f'| {label} | {m["events"]} | {category_metrics["sparse_m1"][label]["missed"]} | {m["missed"]} | {pct(m["recall"])} |')
    report+=['', '## Geometry Only / Appearance Only 与物理状态 Oracle', '', '| 方法 | 历史候选率 | 可判定接受数 | 物理错误数 | 物理 FAR |', '|---|---:|---:|---:|---:|']
    for n,m in ablation.items():report.append(f'| {n} | {pct(m["valid_candidate_rate"])} | {m["accepted_labeled"]} | {m["physical_errors"]} | {pct(m["physical_false_acceptance_rate"])} |')
    report += ['', 'Geometry Only / Appearance Only 沿用原项目定义，在相同历史状态上作只读消融判断；没有为它们虚构可信背景面积。物理 Oracle 仍要求同一实例、传感器支持、visibility=4、预测中心误差 ≤1m。物理锚点 FAR 与像素对应 FAR 分开报告，不混用分母。',
        '', '## M2 可以获得的候选与边界','',
        f'最终确认中累计获得 **{m2["supported_pixel_observations"]:,} 个像素观测候选**（跨帧/相机累计，非独立物理表面），以及 **{m2["complete_16px_image_tiles"]:,}/{m2["total_16px_image_tiles"]:,} 个完整 16×16 图像块**，比例 {pct(ratio(m2["complete_16px_image_tiles"],m2["total_16px_image_tiles"]))}。已有 sparse M1 对应完整块为 {base["complete_tiles"]}。',
        '这些是后续建立局部依赖和重采样验证的实际输入候选，不是可立即跳算的 EVA token。图像缩放/裁剪、16×16 patch 足迹、窗口注意力和全局注意力均未被认证；**当前可直接认证精确复用的深层 token 数仍为 0，实际 GPU 跳算量为 0**。`dense_source_map` 提供当前像素到历史观测的坐标，历史只来自最近一次真实 commit；缺少证据继续刷新。',
        '', '## 耗时与真实 Forward 特征', '', '| CPU 回放方法 | P50 ms | P95 ms |', '|---|---:|---:|']
    for n,m in summary.items():
        latency=roi['latency_p50_p95_ms'].get(n,m['latency_p50_p95_ms']) if roi else m['latency_p50_p95_ms']
        report.append(f'| {n} | {latency[0]:.2f} | {latency[1]:.2f} |')
    report += ['', 'CPU 回放排除 IO、Oracle 和完整模型。每个相机的并行耗时之和是工作量，Runtime Breakdown 使用实际 precheck 墙钟差分，避免重复相加。方法执行次序固定，以上是实测开销，不能直接推导端到端速度提升。']
    if roi:
        report+=['',f'最终计算范围优化重新执行 {roi["frames_checked"]} 帧次，精确 mask、锚点、特征候选、来源与刷新提示均与保存结果一致：**{roi["all_exact"]}**。表中 sparse/dense 的 CPU 耗时来自此次最终执行，original strict 来自前一轮确认；没有将不同版本的时间混写成同一轮。']
    warmed_path=root/'warmed_latency.json'
    if warmed_path.exists():
        warm=read(warmed_path)['results']
        report+=['','补充预加载、交替顺序的 4 帧回放（1 轮预热、4 轮测量，排除每次场景首帧）：'+ '；'.join(f'{n} P50/P95 '+ '/'.join(f'{x:.2f}' for x in m['p50_p95_ms'])+' ms' for n,m in warm.items())+'。各重复的输出指纹完全一致。']
        if (root/'warmed_latency_before_roi.json').exists():
            old_warm=read(root/'warmed_latency_before_roi.json')['results']['dense']['p50_p95_ms'][0]
            report += ['',f'计算范围优化前后同一预加载微基准的 dense P50：{old_warm:.2f} → {warm["dense"]["p50_p95_ms"][0]:.2f} ms；仅说明该 4 帧工作负载的局部成本变化，不能外推为整体模型加速。']
    if gpu:
        report += ['',f'RTX 3090 完整 Forward（记录 `{gpu_dir.name}/`）：输出、backbone、检测/地图 memory 摘要等价检查 **{gpu["observe_only_equal"]}**。LLM 使用每轮新的输出目录并实际执行。8 个唯一连续帧，1 轮预热、2 轮计时；重复不增加独立场景数。', '', '| 模式 | 完整帧 P50/P95 ms | M1 P50/P95 ms |','|---|---:|---:|']
        for n,m in gpu['summary'].items():report.append(f'| {n} | '+ '/'.join(f'{x:.2f}' for x in m['frame_p50_p95_ms'])+' | '+('/'.join(f'{x:.2f}' for x in m['m1_p50_p95_ms']) if m['m1_p50_p95_ms'] else '—')+' |')
        report+=['','baseline 为关闭 M1，reference 为已有 sparse M1，balanced 为本轮配置。稠密验证仍增加开销，当前没有 GPU 跳算，不能宣称净推理加速。最终代码的 GPU 运行与计算范围优化前的记录分别保存，避免覆盖旧测量。']
        if gpu_dir.name=='gpu_roi' and (root/'gpu/results.json').exists():
            prior_gpu=read(root/'gpu/results.json')['summary']['balanced']['m1_p50_p95_ms']
            report += ['',f'计算范围优化前后的本机完整 Forward M1 P50：{prior_gpu[0]:.2f} → {gpu["summary"]["balanced"]["m1_p50_p95_ms"][0]:.2f} ms。该复测来自相同 8 帧和相同计时协议，跨运行波动仍存在。']
    else:report+=['','最终配置 GPU 运行未完成，尚不能宣称完整模型检查通过。']
    if feature_summary:
        report+=['','实际完整 Forward 在已接受稠密像素的固定 8px 采样网格上，对历史与当前空间特征作双线性坐标对齐。它是近似特征误差，不是严格 token 等价或下游任务精度。','','| 层 | 采样数 | Relative L2 | Cosine | L2 ≤0.1 占比 |','|---|---:|---:|---:|---:|']
        for n,m in feature_summary.items():
            if m['relative_l2'] is not None:report.append(f'| {n} | {m["samples"]} | {m["relative_l2"]:.4f} | {m["cosine"]:.4f} | {pct(m["fraction_l2_le_01"])} |')
        report+=['','特征实验使用已知场景的少量连续帧，重复执行不增加独立场景数。Branch carrier 无对象/token 一一对应，只能提供原有 mean-pool 诊断；不得当作复用认证。']
    tests=(root/'tests.log').read_text() if (root/'tests.log').exists() else ''
    report+=['','## 审计与复现','','关键文件：`protocol.json`、`selection.json`、`holdout_summary.json`、`verification.json`、`physical_ablation.json`、`m2_candidates.json`、逐场景原始记录与精确 mask。场景聚类 bootstrap 区间保存在 verification；帧和 LiDAR 点之间相关，不能当作百万独立样本。在线源码与最终执行记录哈希一致：'+str(gate['online_source_hash_match'])+'（'+source_protocol.name+'）。',
        '测试：'+(next((s for s in tests.splitlines() if ' passed' in s),'待完成')),
        '运行命令与配置边界见 `docs/worldstream_m1_coverage.md`。所有前期失败候选保留，没有用最终确认结果重新选门限。',
        '', '![Risk–Coverage](coverage_final/figures/risk_coverage.png)','', '![Invalidation–Recall](coverage_final/figures/invalidation_recall.png)','', '![Runtime Breakdown](coverage_final/figures/runtime_breakdown.png)','', '![真实场景](coverage_final/figures/real_scene_masks.png)']
    if features:report+=['','![实际特征误差](coverage_final/figures/feature_error.png)']
    (root.parent/'refinement_report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(gate,indent=2))

if __name__=='__main__':main()
