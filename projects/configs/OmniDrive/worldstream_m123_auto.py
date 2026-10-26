"""World-referenced stateful VLM serving: the only M1/M2/M3 route."""
_base_ = ['./mask_eva_lane_det_vlm.py']
model = dict(
    worldstream_require_m123=True,
    m1_enable=True, m1_execution_mode='observe_only',
    m1_config='configs/worldstream_m1_final.yaml',
    m2_enable=True,
    m2_config=dict(mode='auto', reuse_boundary='window', state_interface='source_linked',
        m1_service='maintain', memory_budget_mb=512, ordinary_budget=3, max_reuse_frames=8),
    m3_enable=True,
    m3_config=dict(mode='auto', state_context='augment', memory_budget_mb=512,
        reuse_strategy='wsdr_kv',block_size=32,quality_budget=0.02,
        reuse_policy='outputs/final_route/m3_reuse_policy.json'))
# Preserve deployed preprocessing and retain source camera poses in metadata.
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', training=False, data_aug_conf=dict(
        resize_lim=(0.37, 0.45), final_dim=(320, 640), bot_pct_lim=(0., 0.),
        rot_lim=(0., 0.), H=900, W=1600, rand_flip=False)),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='NormalizeMultiviewImage', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadAnnoatationVQATest', base_vqa_path='./data/nuscenes/vqa/val/',
         base_conv_path='./data/nuscenes/conv/val/', base_counter_path='./data/nuscenes/eval_cf/',
         load_type=['planning'], tokenizer='ckpts/pretrain_qformer/', max_length=2048),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='PETRFormatBundle3D', with_label=False,
                  collect_keys=['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
                                'ego_pose', 'ego_pose_inv', 'command', 'can_bus'],
                  class_names=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                               'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
             dict(type='Collect3D', keys=['input_ids', 'img', 'lidar2img', 'intrinsics', 'extrinsics',
                   'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus'],
                  meta_keys=('sample_idx', 'vlm_labels', 'filename', 'ori_shape', 'img_shape',
                             'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                             'img_norm_cfg', 'scene_token', 'cam_infos'))])]
data = dict(test=dict(pipeline=test_pipeline), val=dict(pipeline=test_pipeline))
