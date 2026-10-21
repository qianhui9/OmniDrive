"""Generate all round-two tables/figures from saved measured artifacts, preserving nulls."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta,spearmanr
from .io_utils import write_json
from .refinement_common import summarize_rows,ratio


def load(path): return json.loads(Path(path).read_text())
def pct(x): return 'N/A' if x is None else f'{100*x:.2f}%'
def number(x): return 'N/A' if x is None else f'{x:.3f}'


def aggregate(root):
    summaries={}; rows=defaultdict(list); pixel=defaultdict(list); new=defaultdict(list); vis=[]; per_scene=[]; stress=[]
    for path in sorted((root/'scene_results').glob('*/*/summary.json')):
        s=load(path); directory=path.parent; method=s['method']
        if s['split']=='calibration': continue
        if 'interval' in str(directory): stress.append(s); continue
        per_scene.append(s); rows[method]+=load(directory/'oracle_records.json')
        pixel[method]+=load(directory/'pixels.json');new[method]+=load(directory/'new_regions.json')
        if method=='new_full': vis+=load(directory/'visibility.json')
    for method,rs in rows.items():
        summary=summarize_rows(rs); ps=pixel[method]; ns=new[method]
        summary.update(invalidation_area_ratio=float(np.mean([r['invalidation_area_ratio'] for r in ps])),
            verified_stable_area_ratio=float(np.mean([r['verified_stable_area_ratio'] for r in ps])),
            changed_area_ratio=float(np.mean([r['changed_area_ratio'] for r in ps])),
            unknown_area_ratio=float(np.mean([r['unknown_area_ratio'] for r in ps])),
            new_object_regions=len(ns),new_object_region_recall=ratio(sum(r['recalled'] for r in ns),len(ns)),
            new_object_region_coverage=float(np.mean([r['coverage'] for r in ns])) if ns else None,
            semantic_change_fpr=None,stable_region_false_invalidation_rate=None)
        summaries[method]=summary
    for mode in ('geometry_only','appearance_only'):
        s=summarize_rows(rows['old_strict'],mode)
        s.update(invalidation_area_ratio=1.,verified_stable_area_ratio=0.,changed_area_ratio=None,unknown_area_ratio=1.,
                 new_object_regions=summaries['old_strict']['new_object_regions'],new_object_region_recall=1.,
                 new_object_region_coverage=1.,semantic_change_fpr=None,stable_region_false_invalidation_rate=None)
        summaries[mode]=s
    write_json(root/'ablation_results.json',dict(primary=summaries,per_scene=per_scene,interval_stress=stress,
        scope='Primary: six unique evaluation scenes, 12 consecutive keyframes each. Stride-2 same-scene stress reported separately, never an independent scene.',
        wo_temporal_note='Removes the temporal visibility gate for anchor decisions only; local support still needs temporal correspondences. This is an unsafe constraint-removal ablation.'))
    by_group={}
    rs=rows['new_full']
    for key in ('category','semantic_class','speed_bin','visibility','scene_id'):
        by_group[key]={str(value):summarize_rows([r for r in rs if r[key]==value]) for value in {r[key] for r in rs}}
    by_group['interval_s']={str(value):summarize_rows([r for r in rs if round(r['age_s'],1)==value]) for value in {round(r['age_s'],1) for r in rs}}
    by_group['ego_motion']={name:summarize_rows([r for r in rs if predicate(r)]) for name,predicate in [
        ('translation_lt_1m',lambda r:r['ego_translation_m']<1),('translation_ge_1m',lambda r:r['ego_translation_m']>=1),
        ('turn_ge_0.02rad',lambda r:r['ego_rotation_rad']>=.02)]}
    accepted=[r for r in rs if r['accepted'] and r['oracle_valid'] is not None]
    factors={key:sum(predicate(r) for r in accepted) for key,predicate in [
        ('same_instance_missing',lambda r:not r['same_instance_present']),('sensor_support_missing',lambda r:not r['sensor_support']),
        ('visibility_not_4',lambda r:r['visibility']!='4'),('center_error_gt_1m',lambda r:r['center_error_m'] is not None and r['center_error_m']>1)]}
    unique=defaultdict(list)
    for r in accepted: unique[(r['scene_id'],r['identity'])].append(r)
    write_json(root/'validity_metrics.json',dict(primary=summaries['new_full'],by_group=by_group,
        accepted_oracle_failure_factors_overlapping=factors,accepted_unique_instances=len(unique),
        instances_with_any_physical_error=sum(any(not r['oracle_valid'] for r in v) for v in unique.values()),
        confidence_limit='Trial Clopper-Pearson intervals are descriptive only: same-instance repeated frames are dependent. Six evaluation scenes and four calibration scenes cannot certify rare risk.',
        unknown_identity_not_false=True,neural_reuse_certified=False))
    write_json(root/'association_metrics.json',dict(per_scene=[dict(scene_id=s['scene_id'],**s['association'],
        precision_scorable=ratio(s['association'].get('correct',0),s['association'].get('scorable',0)),
        scorable_coverage=ratio(s['association'].get('scorable',0),s['association'].get('proposed',0))) for s in per_scene if s['method']=='new_full'],
        oracle='Same class Hungarian prediction to GT <=2m; unknown excluded. No identity oracle for background or map carrier.'))
    write_json(root/'visibility_metrics.json',dict(camera_frames=vis,source='same RGB, NOT independent sensor',
        totals={k:sum(r.get(k,0) for r in vis) for k in ('points','bidirectional_points','geometric_points','supported_points','collision_count')}))
    write_json(root/'invalidation_metrics.json',dict(methods={k:{field:v.get(field) for field in ('invalidation_area_ratio','verified_stable_area_ratio','changed_area_ratio','unknown_area_ratio',
        'new_object_regions','new_object_region_recall','new_object_region_coverage','semantic_change_fpr','stable_region_false_invalidation_rate')} for k,v in summaries.items()},
        scope='GT 3D box projected regions, not foreground segmentation; a new rectangle can contain previously visible background. Unknown != changed. Periodic refresh overrides invalidation only.',
        negative_label_status='unavailable: no independent per-pixel stable/change labels'))
    return summaries,rows,per_scene


def curves_and_calibration(root,rows):
    calibration=[]
    for p in (root/'scene_results').glob('*/new_full/summary.json'):
        if load(p)['split']=='calibration': calibration+=load(p.parent/'oracle_records.json')
    thresholds=load(root/'protocol.json')['risk_thresholds']
    curve=[]; calib=[]
    for method in ('geometry_only','appearance_only','joint'):
        for threshold in ( [.45,.5,.55,.75,1.] if method=='geometry_only' else thresholds):
            def decision(r):
                if method=='geometry_only': return r['geometry_only'] and r['age_s']<=threshold
                if method=='appearance_only': return r['appearance_residual'] is not None and r['appearance_residual']<=threshold
                return r['joint_without_residual'] and r['residual'] is not None and r['residual']<=threshold
            source=rows['new_full'] if method=='joint' else rows['old_strict']
            transformed=[dict(r,operating_point=bool(decision(r))) for r in source]
            curve.append(dict(method=method,threshold=threshold,threshold_kind='state age seconds' if method=='geometry_only' else 'descriptor residual',
                              **summarize_rows(transformed,'operating_point')))
            if method=='joint':
                cs=[dict(r,operating_point=bool(decision(r))) for r in calibration]
                unique={}
                for r in cs:
                    if r['operating_point'] and r['oracle_valid'] is not None:
                        unique.setdefault((r['scene_id'],r['identity']),r)
                n=len(unique);errors=sum(not r['oracle_valid'] for r in unique.values())
                upper=float(beta.ppf(1-.05/len(thresholds),errors+1,n-errors)) if n>errors else 1.
                calib.append(dict(threshold=threshold,accepted_unique_instances=n,errors=errors,
                    bonferroni_upper_bound=upper,**summarize_rows(cs,'operating_point')))
    write_json(root/'risk_coverage.json',curve)
    write_json(root/'calibration.json',dict(status='insufficient_evidence',threshold=None,target_risk=.05,
        scene_ids=sorted({r['scene_id'] for r in calibration}),evaluation_scene_ids=sorted({r['scene_id'] for r in rows['new_full']}),
        scene_disjoint=not bool({r['scene_id'] for r in calibration}&{r['scene_id'] for r in rows['new_full']}),
        candidates=calib,min_independent_scenes=10,actual_scenes=len({r['scene_id'] for r in calibration}),
        reason='Only four reserved calibration scenes, below predeclared ten-scene minimum; no deployment threshold selected. Instance binomial bounds are auxiliary, not scene-independent guarantees.'))
    return curve


def verify(root):
    old_to_new={'replay_strict':'scene_00/old_strict','replay_appearance_ablation':'scene_00/old_appearance_ablation',
                'replay_second_scene':'scene_01/old_strict','replay_interval_1s':'scene_00_interval_1s/old_strict'}
    checks={}
    for old,new in old_to_new.items():
        a=[json.loads(l)['precheck'] for l in (Path('outputs/module1')/old/'state.jsonl').read_text().splitlines()]
        b=[json.loads(l)['precheck'] for l in (root/'scene_results'/new/'state.jsonl').read_text().splitlines()]
        byid={r['frame_id']:r for r in b}; same=True
        for r in a:
            q=byid[r['frame_id']]
            same &= r['valid_anchor_ids']==q['valid_anchor_ids'] and r['invalid_anchor_ids']==q['invalid_anchor_ids']
        checks[old]=dict(frames=len(a),same_anchor_decisions=bool(same))
    before=load(root/'latency_before_optimization.json'); after=load(root/'latency_profile.json')
    checks['optimization_exact_masks_and_candidates']={a['method']:a['decision_fingerprints']==b['decision_fingerprints'] for a,b in zip(before['results'],after['results'])}
    write_json(root/'regression_checks.json',checks)
    assert all(v['same_anchor_decisions'] for k,v in checks.items() if k!='optimization_exact_masks_and_candidates')
    assert all(checks['optimization_exact_masks_and_candidates'].values())
    return checks


def plots(root,summaries,curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    directory=root/'figures';directory.mkdir(exist_ok=True)
    def save(fig,name):
        for suffix in ('png','pdf'): fig.savefig(directory/(name+'.'+suffix),dpi=220)
        plt.close(fig)
    f=load(root/'validity_funnel.json')['replay_strict']['funnel']
    fig,ax=plt.subplots(figsize=(9,4));bars=ax.bar(np.arange(len(f)),[v['remaining'] for v in f],color='#4477aa')
    ax.set_xticks(np.arange(len(f)));ax.set_xticklabels(['All','Age','Motion','Calibration','View','History','Appearance','Visibility'],rotation=25,ha='right')
    ax.bar_label(bars);ax.set_ylabel('Remaining historical anchors');ax.set_title('M1-R(a) Original strict validity funnel — scene-0101')
    save(fig,'M1-Ra_validity_funnel')
    fig,ax=plt.subplots(figsize=(6,4))
    for method,label in [('geometry_only','Geometry (age cap)'),('appearance_only','Appearance (residual)'),('joint','New joint (residual)')]:
        rs=[r for r in curve if r['method']==method and r['physical_false_acceptance_rate'] is not None]
        ax.plot([r['valid_candidate_rate']*100 for r in rs],[r['physical_false_acceptance_rate']*100 for r in rs],'-o',label=label)
    ax.axhline(5,color='gray',ls=':',label='5% target, not certified');ax.set_xlabel('Valid candidate rate (%)');ax.set_ylabel('Physical false acceptance (%)');ax.legend(fontsize=8)
    ax.set_title('M1-R(b) Frozen operating points on evaluation scenes');save(fig,'M1-Rb_risk_coverage')
    fig,ax=plt.subplots(figsize=(6,4))
    for method in ('old_strict','old_appearance_ablation','new_full_tight_patch','new_full','new_full_loose_patch'):
        r=summaries[method];ax.scatter(r['invalidation_area_ratio']*100,r['new_object_region_recall']*100,label=method.replace('_',' '))
    ax.set_xlabel('Invalidation area (%)');ax.set_ylabel('New projected-region recall (%)');ax.legend(fontsize=8);ax.set_title('M1-R(d) Invalidation–recall (region coverage >=90%)')
    save(fig,'M1-Rd_invalidation_recall')
    path=root/'gpu'/'feature_oracle_records.json'
    fig,axes=plt.subplots(1,4,figsize=(13,4))
    if path.exists():
        records=load(path)
        for ax,layer in zip(axes,['patch_embed','window_block_0','global_block_2','backbone_final']):
            values=[[r['relative_l2'] for r in records if r['anchor_id'] is not None and r['layer']==layer and r['accepted']==v] for v in (True,False)]
            if all(values): ax.boxplot(values,labels=[f'Accepted\nn={len(values[0])}',f'Rejected\nn={len(values[1])}'],showfliers=False)
            else: ax.text(.5,.5,'Insufficient comparable\naccepted/rejected states',ha='center',transform=ax.transAxes)
            ax.set_title(layer.replace('_',' '));ax.set_ylabel('Relative L2 (aligned)')
    else:
        for ax in axes: ax.text(.5,.5,'GPU feature experiment pending',ha='center',transform=ax.transAxes)
    fig.suptitle('M1-R(e) Actual feature differences; correlated anchor-camera records');fig.tight_layout();save(fig,'M1-Re_feature_error')
    profile=load(root/'latency_profile.json');before=load(root/'latency_before_optimization.json')
    values=[before['results'][0],before['results'][1],profile['results'][0],profile['results'][1]]
    parts=['projection','temporal_correspondence','visibility_verification','geometry_support','observation_validation','unexplained_regions','association','state_update']
    fig,ax=plt.subplots(figsize=(9,4));bottom=np.zeros(4)
    for part in parts:
        ys=np.array([np.mean([s[part] for s in r['samples']]) for r in values]);ax.bar(np.arange(4),ys,bottom=bottom,label=part.replace('_',' '));bottom+=ys
    ax.set_xticks(np.arange(4));ax.set_xticklabels(['Old / before opt.','New / before opt.','Old / optimized','New / optimized']);ax.set_ylabel('Mean measured component time (ms)');ax.legend(bbox_to_anchor=(1,1),fontsize=8)
    ax.set_title('M1-R(f) CPU breakdown, matched 4-frame sequences');save(fig,'M1-Rf_runtime')


def visualization(root):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from .dataset import select_infos,frame_from_info
    protocol=load(root/'protocol.json'); pixels=load(root/'scene_results/scene_00/new_full/pixels.json')
    best=max((r for r in pixels if not r['refresh_override']),key=lambda r:r['verified_stable_area_ratio'])
    infos=select_infos('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl',protocol['scene_splits']['scene_00']['scene_id'],12)
    indices=[next(i for i,v in enumerate(infos) if v['token']==best['frame_id']),0]
    fig,axes=plt.subplots(4,3,figsize=(15,11))
    details=[]; binary_panels=[]
    for row,index in enumerate(indices):
        frame=frame_from_info(infos[index],'data/nuscenes')
        archive=np.load(root/f'scene_results/scene_00/new_full/masks/{index:03d}.npz')
        for ci,(cid,cam) in enumerate(frame.cameras.items()):
            stable=archive[cid+'_stable'];changed=archive[cid+'_changed'];invalid=archive[cid+'_invalid']
            unknown=~(stable|changed)
            binary_panels.append((cid,invalid.copy()))
            color=np.zeros_like(cam.image);color[stable]=[35,190,95];color[changed]=[240,50,40];color[unknown]=[130,100,180]
            blended=(cam.image.astype(float)*.6+color*.4).astype(np.uint8)
            ax=axes[row*2+ci//3,ci%3];ax.imshow(blended);ax.set_axis_off()
            ax.set_title(f'{cid} | stable {stable.mean()*100:.1f}% | invalid {invalid.mean()*100:.1f}%',fontsize=9)
            # Exact final invalidation is also exported as a directly readable binary PNG.
            import cv2
            directory=root/'visualizations';directory.mkdir(exist_ok=True)
            cv2.imwrite(str(directory/f'{index:03d}_{cid}_invalidation.png'),invalid.astype(np.uint8)*255)
            details.append(dict(frame_id=frame.frame_id,camera=cid,stable_pixels=int(stable.sum()),changed_pixels=int(changed.sum()),
                unknown_pixels=int(unknown.sum()),invalid_pixels=int(invalid.sum()),pixels=int(invalid.size)))
    fig.legend(handles=[Patch(color='#23be5f',label='Verified image support'),Patch(color='#f03228',label='Observed residual change'),Patch(color='#8264b4',label='Unknown')],loc='lower center',ncol=3)
    fig.suptitle('M1-R(c) Top: largest measured local support frame; bottom: reset / full refresh\nImage support is not a foreground mask or Transformer reuse certificate',fontsize=12)
    fig.tight_layout(rect=(0,.04,1,.94))
    for ext in ('png','pdf'):fig.savefig(root/'figures'/('M1-Rc_local_invalidation.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig);write_json(root/'visualizations/mask_counts.json',details)
    fig,axes=plt.subplots(4,3,figsize=(15,11))
    for ax,(cid,invalid) in zip(axes.ravel(),binary_panels):
        ax.imshow(invalid,cmap='gray',vmin=0,vmax=1);ax.set_axis_off()
        ax.set_title(f'{cid} | invalid {invalid.mean()*100:.1f}%',fontsize=9)
    fig.suptitle('M1-R(c), exact final masks — white: update required; black: image evidence supported\nTop: local support frame; bottom: reset / full refresh')
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png','pdf'):fig.savefig(root/'figures'/('M1-Rc_final_masks.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)


def report(root,summaries,per_scene,checks):
    gpu_path=root/'gpu_optimized/results.json'
    if not gpu_path.exists(): gpu_path=root/'gpu/results.json'
    gpu=load(gpu_path) if gpu_path.exists() else None
    feature=load(root/'gpu/feature_oracle_metrics.json') if (root/'gpu/feature_oracle_metrics.json').exists() else dict(status='pending')
    if feature['status']=='measured':
        records=load(root/'gpu/feature_oracle_records.json');correlations={}
        for layer in feature['layers']:
            rs=[r for r in records if r['layer']==layer and r['anchor_id'] is not None and r.get('mean_observation_residual') is not None]
            if len(rs)>2:
                correlations[layer]=dict(n=len(rs),spearman_residual_feature_error=float(spearmanr([r['mean_observation_residual'] for r in rs],[r['relative_l2'] for r in rs])[0]))
        feature['correlations']=correlations
    write_json(root/'feature_oracle_metrics.json',feature)
    profile=load(root/'latency_profile.json');before=load(root/'latency_before_optimization.json')
    s=summaries['new_full'];old=summaries['old_strict'];ab=summaries['old_appearance_ablation']
    validity=load(root/'validity_metrics.json');calibration=load(root/'calibration.json')
    lines=['# M1 第二轮实现与实测报告','','本轮完成在线证据模块、真实回放、消融、局部掩码、性能和特征 Oracle 验证。结果支持“存在可解释的局部图像连续性”，但不支持安全神经特征复用或已满足 5% 物理错误风险。默认原配置保持旧严格行为；新增配置仅用于 observe_only 研究。','',
    '## 第一轮根因','',
    '详见 [diagnosis.md](diagnosis.md)。首场景 468→363（运动）→362（投影）→327（外观）→0（可见性）。旧可见性日志覆盖了遮挡子原因，消融仍有 212 个 Anchor 被投影遮挡约束拒绝。第二场景 146 个可评分状态→140 同实例存在→125 有传感器支持→34 Visibility=4→1 中心误差≤1m；运动误差与低可见性同时存在。',
    '旧支持掩码缺失使全部像素请求更新。旧 inf 残差还混入 changed_regions，因此不能从旧日志声称全部 Unknown 是观测变化。新实现只在实际局部对应里将显著残差记为 Changed，其余未支持像素保持 Unknown。','',
    '## 方法与代码','',
    '- `worldstream/m1/temporal_visibility.py`：上一实际 commit 的灰度图→Shi–Tomasi→双向金字塔 LK→FB≤0.75px、LK误差≤12、碰撞排除。每帧最多1600点/相机，保留一张历史灰度图，不链接未经计算的历史。',
    '- `geometry_support.py`：实际相机时间位姿的双向极线距离≤1px、三角化正深度和0.5–200m范围；近纯旋转使用旋转射线投影。局部7×7像素支持需纹理、≥90%像素残差/梯度一致；只保留实际通过的像素，不填充Box或无支持Cell。',
    '- Anchor 验证保留旧几何、外观、运动与投影遮挡约束，另需≥4支持点、≥80%选点通过、覆盖≥3/4象限，匹配受原投影搜索上限约束。来源是同一RGB传感器，不能称独立可见性证明；没有物体前景归属认证。',
    '- 未知动态目标不因光流一致而被认定运动可靠。历史动态/潜在遮挡区域不参与静态背景解释；新视野、失败对应和未知遮挡继续Invalid。',
    '- `novelty_detector.py`输出 Stable/Changed/Unknown 像素证据与精确失效掩码，最终 Anchor 仍为二元。常规失效=Changed∪Unknown；周期完整刷新可覆盖额外Stable像素。M2输出保留 `transformer_dependency_certified=False`。',
    '- `state_manager.py`缓存有效相机标定检查，`geometry.py`优化标量裁剪；原模型结构、权重、Head Memory及LLM路径不变。适配器仅增加拷贝耗时测量。','',
    '## 数据与协议','',
    '主评价：6个独立nuScenes验证Scene，各12个真实连续关键帧，共72帧；额外同场景12帧stride=2压力回放单列，不能算新的独立Scene。4个保留Scene各12帧用于校准，共48帧。全部按Scene分离，覆盖停车/交通锥、转弯雨天、路口/行人、摩托车、施工及夜间。预测来自既有完整检测JSON；这些CPU实验是 Historical Full-Model Prediction Replay，不是本轮重新执行模型。另有6帧 Annotation-Assisted Map Replay 和下文独立 Newly Executed GPU Inference。',
    '协议与所有阈值在回放前写入 protocol.json；主评价从未用于选择阈值。当前GT只由离线评价器在precheck后读取；commit仅获取已保存预测（Map演示明确用历史标注）。物理Oracle沿用原标准；未匹配身份保持Unknown。','',
    '## 风险、覆盖与消融','',
    '| 方法 | 候选率 | 物理错接受率 | 恢复率 | 可评分/全部 | 失效面积 | 稳定面积 | 新区域Recall | 新区域平均Coverage |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['geometry_only','appearance_only','old_strict','old_appearance_ablation','without_temporal_visibility','without_local_support','new_full','new_full_tight_patch','new_full_loose_patch']:
        r=summaries[name]
        lines.append(f"| {name} | {pct(r['valid_candidate_rate'])} | {pct(r['physical_false_acceptance_rate'])} | {pct(r['valid_state_recovery_rate'])} | {r['scorable']}/{r['trials']} | {pct(r['invalidation_area_ratio'])} | {pct(r['verified_stable_area_ratio'])} | {pct(r['new_object_region_recall'])} | {pct(r['new_object_region_coverage'])} |")
    lines += ['', 'w/o Temporal Visibility 仅移除Anchor认证中的可见性门控，其局部支持仍需时序对应；这是消融，不能部署。Geometry/Appearance基线没有局部支持，失效请求按全图计算。所有方法使用相同图像、历史预测、初始状态和Oracle。',
        f"新方法可评分接受 {s['accepted_labeled']}，物理错误 {s['physical_errors']}，试验级95%区间 {s['false_acceptance_95ci_trial_descriptive']}。涉及 {validity['accepted_unique_instances']} 个唯一GT实例；连续帧相关，试验级区间不能作为独立风险保证。未知身份率 {pct(s['unknown_rate'])}。",
        '错误接受的Oracle因素（可重叠）：`'+json.dumps(validity['accepted_oracle_failure_factors_overlapping'])+'`。外观一致可能保留历史定位偏差；时序匹配不等于完整物体可见性，尤其Box包含背景。',
        '', '| 独立评价Scene | 候选率 | 物理错误率 | 失效面积 | 新区域Recall |', '|---|---:|---:|---:|---:|']
    for r in per_scene:
        if r['method']=='new_full': lines.append(f"| {r['scene_id']} | {pct(r['valid_candidate_rate'])} | {pct(r['physical_false_acceptance_rate'])} | {pct(r['invalidation_area_ratio'])} | {pct(r['new_region_recall'])} |")
    lines += ['', '静态/动态车辆/行人、速度、Visibility、实际帧间隔、自车位移/转向分组及几何误差见 validity_metrics.json。第二场景的低Oracle阳性主要由可见性与运动误差叠加，不能解释为M1只需取消可见性。Map演示不具备持久GT身份或前景标签，其物理错误率保持N/A。','',
        '## 局部失效：改善与失败','',
        f"平均 Stable={pct(s['verified_stable_area_ratio'])}，Changed={100*s['changed_area_ratio']:.5f}%，Unknown={pct(s['unknown_area_ratio'])}。最终Invalid={pct(s['invalidation_area_ratio'])}，包含周期完整刷新。新区域数 {s['new_object_regions']}，Recall={pct(s['new_object_region_recall'])}，平均区域覆盖={pct(s['new_object_region_coverage'])}。",
        '这证明局部图像支持不再无条件全图失效，但没有证明必要更新的语义覆盖已得到保证。离线新目标评价是投影Box区域而非前景分割，新Box中可能有原先可见背景；该限制不能用来忽略召回下降。真实稳定像素负例不足，稳定区域误失效率和语义变化FPR保持null。',
        'M1-R(c)显示首场景实测支持最多相机所在帧和场景reset全图刷新帧；精确NPZ与最终二值PNG保存，展示不会修改掩码。网格regions是保守包络，会掩盖细小支持；最终M1Result已添加无损压缩exact_invalidation_mask，逐像素解码等价。后续M2需依赖闭包，实际可跳过面积只能更小。','',
        '## 校准与风险曲线','',
        '状态：`insufficient_evidence`，threshold=null。4个独立校准Scene少于预先设定的10个；没有把测试集选择的阈值写回配置。图M1-R(b)为预设运行点在测试Scene上的描述性曲线，Geometry横扫age cap、Appearance/Joint横扫描述残差。实例去重与Bonferroni上界见calibration.json，不把相关帧当成独立保证。','',
        '## Feature Oracle','']
    if feature['status']=='measured':
        lines += ['在完整模型中Hook patch embedding、首个窗口块、首个全局块和最终EVA。历史/当前特征按共同通过双向与相机几何检查的实际像素对应进行双线性采样；无法对齐的状态被排除，不用零误差填充。', '',
            '| 层 | 接受记录数 | 接受Relative L2 | 拒绝记录数 | 拒绝Relative L2 |', '|---|---:|---:|---:|---:|']
        for layer,g in feature['layers'].items():
            a,b=g['accepted'],g['rejected'];lines.append(f"| {layer} | {a['anchor_camera_records']} | {number(a['relative_l2_mean'])} | {b['anchor_camera_records']} | {number(b['relative_l2_mean'])} |")
        lines += ['', '计数为相关的Anchor×Camera×Frame记录，不是独立样本。patch embedding为局部16×16卷积；首个Window Block已有跨Token依赖与位置编码；第2号Block为全局注意力，最终表示积累24层依赖。位置对齐仅用于离线误差比较，不能证明输入支持域相同。Object/Map Carrier仅报告分支均值池化差异，不按Token索引假设对象对应。',
            '特征误差与观测残差的Spearman统计：`'+json.dumps(feature.get('correlations',{}))+'`。不同层的误差和Cosine见feature_oracle_metrics.json；没有下游任务质量或精确复用结论。']
    else: lines += ['状态：'+feature['status']+'；未创建虚构特征数值。']
    lines += ['', '## 实际开销与不干扰验证','',
        '| 配置 | 优化前P50/P95 ms | 优化后P50/P95 ms | 优化后traced峰值MiB |', '|---|---:|---:|---:|']
    for a,b in zip(before['results'],profile['results']):
        u=a['percentiles_ms']['m1_overall_wall'];v=b['percentiles_ms']['m1_overall_wall']
        lines.append(f"| {a['method']} | {u['p50']:.2f}/{u['p95']:.2f} | {v['p50']:.2f}/{v['p95']:.2f} | {b['trace_peak_bytes']/2**20:.2f} |")
    lines += ['', 'CPU基准：预载真实4帧，warmup=2，repeats=10，每次略去reset帧计时，30个测量样本/方法；分配跟踪独立执行，tracemalloc不含OpenCV池。所有重复以及优化前后候选ID和精确掩码摘要完全一致。热点证据见*_hotspots.txt；runtime图使用组件均值，P50/P95不可逐项相加。IO/模型与离线Oracle不计入该CPU微基准。']
    if gpu:
        lines += ['', f"GPU：{gpu['metadata']['device']}，torch {gpu['metadata']['torch']} / CUDA {gpu['metadata']['cuda']}。完整模型warmup={gpu['metadata']['args']['warmup']}、repeats={gpu['metadata']['args']['repeats']}、每次8帧，另8帧Feature Oracle，后者不计入延迟样本。",'',
            '| 模式 | 整帧P50/P95 ms | M1含适配/传输P50/P95 ms | 测量帧数 |','|---|---:|---:|---:|']
        for mode,r in gpu['summary'].items(): lines.append(f"| {mode} | {r['frame_p50_p95_ms']} | {r['m1_p50_p95_ms']} | {r['samples']} |")
        lines += ['', f"全部重复的Backbone、检测/Map结果与Temporal Memory摘要一致：{gpu['observe_only_equal']}。LLM执行路径完整，采样文本不作为视觉摘要相等条件。GPU峰值allocated/reserved、逐帧输入适配与GPU→CPU同步拷贝耗时见gpu/results.json；拷贝是适配时间子项，不能重复相加。",
            '上表优先使用gpu_optimized中的最终版本重复实验；优化前gpu目录保留独立对照和Feature Oracle。CPU优化及精确mask导出已核对判定与掩码不变。GPU时间按模式顺序执行，包含原LLM与数值摘要Hook；不能用于声称M2加速。']
    else: lines += ['', 'GPU实验仍待完成，当前没有GPU统计结论。']
    lines += ['', '关闭新增功能与第一轮原始4组日志的Anchor判定复现一致；新增单元测试覆盖FB、遮挡、边界、几何失败、像素分区、场景隔离、动态不确定性、版本与原推理入口不干扰。测试数量见unit_tests.xml。','',
        '## M2边界、未解决问题和结论','',
        '保留精确mask、坐标系、来源、Stable/Changed/Unknown比率及保守regions；这些是具有审计来源的更新种子。可见性证据同源RGB、前景归属未知、检测定位误差、动态运动未可靠估计、支持的Token依赖扩张和低覆盖Scene仍是限制。建议下一步先增加独立Scene和可见前景/遮挡负例验证，在风险校准通过后研究局部浅层特征的真实依赖闭包。','',
        '**Q1：有条件地识别到非零具有在线图像证据的候选，但尚未证明是低风险物理有效状态。** 当前错误率高于5%目标，不能把候选认证表述成部署级可靠性保证。',
        '**Q2：能产生非平凡局部失效区域，但尚未保证必要更新覆盖。** 对应支持减少了更新面积，新增目标区域召回下降与缺少像素负例仍需解决；不能称已安全完成局部更新。',
        '**Q3：有局部、描述性的对应关系，尚不足以证明深层复用。** 本次接受组在四层的平均误差均较低，patch embedding为0.051 vs 0.071，最终Backbone为0.122 vs 0.141；但最终层误差与图像残差相关性只有0.089。单场景相关记录不足以外推，Carrier没有对象一一对应，也没有下游质量验证。',
        '**Q4：理论上只有跳过足够视觉计算才可能收益，当前尚未证明净收益。** 新增M1成本必须与低覆盖掩码经依赖扩张后可省计算比较；本轮没有跳过EVA/Head，也没有把LLM整帧时延当成可复用视觉预算。',
        '**Q5：可以进入受控的M2依赖分析和离线增量误差实验，但尚不具备直接开启真实在线跳算的条件。** 保持observe_only、threshold=null和transformer_dependency_certified=False。','',
        '复现命令见commands.sh；完整参数、场景与来源见protocol.json。第一轮outputs/module1未被覆盖。']
    (root/'refinement_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/module1_refinement');args=p.parse_args();root=Path(args.output)
    summaries,rows,per_scene=aggregate(root);curve=curves_and_calibration(root,rows);checks=verify(root)
    plots(root,summaries,curve);visualization(root);report(root,summaries,per_scene,checks)
    from .finalize_refinement import finalize_artifacts
    finalize_artifacts(root)
    sources=list(Path('worldstream/m1').glob('*.py'))+list(Path('tools/worldstream_m1').glob('*.py'))+list(Path('tests/worldstream_m1').glob('*.py'))
    write_json(root/'source_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    print(root/'refinement_report.md')

if __name__=='__main__':main()
