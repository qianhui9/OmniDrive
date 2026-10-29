# WRSAR v4

Research verdict **NO_GO**; best DEV policy target_dev_0, target completeness enabled=False. 969 actual full M123 calls; 77 formal actual skips.

P0: correct low-score pedestrian candidates are filtered at M1 .3 or M3 .35, while the category case retains its motorcycle but selects insufficiently supported near Car. Detection truth and nearest-target truth differ. The opt-in readout reuses existing CPU decoder scalars before the unchanged M1 filter; immutable competition evidence never becomes an Answer Witness. Current detector-center evidence remains distinct from M2 reuse qualification.

New TEST four disjoint experimental logs/32 frames, historical scene/log metadata excluded. Source partition is physical train; possible base-model training exposure is disclosed. Addressable3Q 1.344391x; mixed3Q 1.116378x. Paired quality B−C=-0.3170731707317073, sceneCI=[-0.5833333333333334, -0.16666666666666666], conservative all-request bounds=[-0.2604166666666667, -0.010416666666666666]; new severe=1 outputs/1 unique cases. Independent target-completeness value=False.

All thresholds, grading sources, average/severe research criteria were frozen before TEST. One runtime/generator, unchanged weights/sampling/RNG. Production options remain off; deployment safety is NOT_QUALIFIED. Historical v3 NO_GO is unchanged.


## Acceptance outcome and scope

Final **NO_GO**. DEV selected target_dev_0 (target enabled=False), not an unmeasured post-TEST alternative. Research average-quality budget passes=True; credible object3Q E2E gain=True; the predeclared research severe gate passes=False.

Object3Q B→C=1.344391x; B known quality=58.536585%, C=90.243902%, coverage=41/96, UNKNOWN=55. New severe rate=2.439024%, empirical sceneCI=[0.0, 0.08333333333333333], frozen ceiling2%. Overall severe outputs B=17, C=4; lower totals do not erase the newly introduced case.

- `2ea0bbd8c3ca442baa3f6c0db66c087d` Where is the nearest pedestrian relative to the vehicle?: B='The nearest pedestrian is on the sidewalk to the right-front of the vehicle.', C='The nearest pedestrian is to the front-left.'; reference direction=['front', 'right'], center=[19.766369359234147, -2.404142263128165, 1.085897511310396]; selected current fact=[{'anchor_id': '12cd5f88732749a58725528bb3d86958:g147:a3', 'semantic_class': 'pedestrian', 'center_ego': [3.194983742366958, 18.60588041610265, 1.5794973085707085], 'directions': ['front', 'left'], 'motion': '', 'velocity_ego': [], 'confidence': 0.46821796894073486, 'cameras': ['CAM_BACK_LEFT', 'CAM_FRONT_LEFT'], 'source_frame': '2ea0bbd8c3ca442baa3f6c0db66c087d', 'source_timestamp': 1533114559.147787, 'state_version': 8, 'state_epoch': 147, 'evidence_origin': 'actual_perception', 'identity_qualified': False, 'observation_kind': 'current_photometric_support', 'identity_required': False, 'association_unambiguous': False}].

Measured independent target quality/severe value=False. Core ablation vetoes={}; equal skip decisions=True, same-token pairs=224/224. Readout/check overhead and phase/GPU variation can coexist; timing differences with the same compute receipts do not establish new language-compute savings. Old diagnostic replays and DEV coverage remain separate; TEST never selects alternative configurations.

Coverage by original workload: addressable1Q: structured=32/32, supported=14, actual skips=11; addressable3Q: structured=96/96, supported=60, actual skips=47; fixed1Q: structured=0/32, supported=0, actual skips=0; fixed3Q: structured=0/96, supported=0, actual skips=0; mixed3Q: structured=33/96, supported=21, actual skips=18; natural1Q: structured=0/32, supported=0, actual skips=0; natural3Q: structured=1/96, supported=1, actual skips=1. The mixed workload adds one preset structured slot; additional original questions can also parse as structured. It is not a natural population estimate. Natural1Q has a small measured E2E regression; its module breakdown/sceneCI remain visible, and the whole change cannot be assigned to microsecond parsing.

Only4 independent experimental TEST logs/32 frames are available, with reliable object3Q labels on3 logs. Source partition is physical train; possible base-model training exposure is explicit. Rare-error and UNKNOWN uncertainty prevent a deployment-safety claim. Risk ratios are descriptive, not probabilities. v3 historical NO_GO is unchanged; production switches stay off.

The single post-inference report-key collision was fixed with an exact recorded metadata substitution. Frozen original source and policies are preserved; inference and scoring logic are unchanged. See report_assembly_fix.json and integrity_audit.json.
