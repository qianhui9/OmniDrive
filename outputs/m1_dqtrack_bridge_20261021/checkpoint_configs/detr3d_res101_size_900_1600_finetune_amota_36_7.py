point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesTrackDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    cam_sweep_num=1)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewMultiSweepImageFromFiles',
        sweep_num=1,
        to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='TrackletRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32)
]
test_pipeline = [
    dict(
        type='LoadMultiViewMultiSweepImageFromFiles',
        sweep_num=1,
        to_float32=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32)
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type='NuScenesTrackDataset',
        data_root='data/nuscenes/',
        ann_file='data/infos/track_cat_10_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewMultiSweepImageFromFiles',
                sweep_num=1,
                to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='TrackletRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32)
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            cam_sweep_num=1),
        test_mode=False,
        box_type_3d='LiDAR',
        num_frames_per_sample=3,
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(
                type='CollectUnified3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img',
                    'timestamp'
                ])
        ],
        track_classes=[
            'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
            'pedestrian'
        ],
        force_continuous=True,
        use_valid_flag=True),
    val=dict(
        type='NuScenesTrackDataset',
        data_root='data/nuscenes/',
        ann_file='data/infos/track_cat_10_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewMultiSweepImageFromFiles',
                sweep_num=1,
                to_float32=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32)
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            cam_sweep_num=1),
        test_mode=True,
        box_type_3d='LiDAR',
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(type='CollectUnified3D', keys=['img', 'timestamp'])
        ],
        track_classes=[
            'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
            'pedestrian'
        ],
        num_frames_per_sample=1),
    test=dict(
        type='NuScenesTrackDataset',
        data_root='data/nuscenes/',
        ann_file='data/infos/track_cat_10_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewMultiSweepImageFromFiles',
                sweep_num=1,
                to_float32=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32)
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            cam_sweep_num=1),
        test_mode=True,
        box_type_3d='LiDAR',
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(type='CollectUnified3D', keys=['img', 'timestamp'])
        ],
        track_classes=[
            'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
            'pedestrian'
        ],
        num_frames_per_sample=1))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadMultiViewMultiSweepImageFromFiles',
            sweep_num=1,
            to_float32=True),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32)
    ])
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/detr3d_res101_size_900_1600_fcos3d_dq_chn128_em_2_temp_02_uniconv_3_cont_aug_pro_cate_10_mf_decay_0_7_fp_0_2_lr_1e5_track_only_e5'
load_from = 'ckpts/pretrain/detr3d_resnet101_cbgs.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.2, 0.2, 8]
cam_sweep_num = 1
fp16_enabled = True
bev_stride = 4
sample_num = 11
track_frame = 3
voxel_shape = [128, 128, 11]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
track_names = [
    'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian'
]
velocity_error = [6, 6, 6, 5, 12, 5, 5]
model = dict(
    type='DETR3DTrackerDualQuery',
    use_grid_mask=False,
    tracker_cfg=dict(
        track_frame=3,
        num_track=300,
        num_query=900,
        num_cams=6,
        num_sweeps=1,
        embed_dims=256,
        ema_decay=0.5,
        train_det_only=False,
        train_track_only=True,
        class_dict=dict(
            all_classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            track_classes=[
                'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
                'pedestrian'
            ]),
        pos_cfg=dict(
            update_pos=True,
            update_all=True,
            detach_pos=True,
            detach_velo=True,
            pos_trans='ffn',
            fuse_type='sum',
            final_trans='linear'),
        query_trans=dict(
            with_att=True, with_pos=True, min_channels=256, drop_rate=0.0),
        track_aug=dict(drop_prob=0, fp_ratio=0.2),
        ema_drop=0.0,
        det_embed_grad=True,
        update_before=False,
        class_spec=True,
        eval_det_only=True,
        velo_error=[6, 6, 6, 5, 12, 5, 5],
        assign_method='hungarian',
        det_thres=0.2,
        new_born_thres=0.2,
        asso_thres=0.1,
        miss_thres=7),
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='Detr3DHead',
        num_query=900,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='Detr3DTransformer',
            decoder=dict(
                type='Detr3DTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAttenRaw',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            num_points=1,
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=300),
    loss_cfg=dict(
        type='DualQueryMatcher',
        num_classes=10,
        class_dict=dict(
            all_classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            track_classes=[
                'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
                'pedestrian'
            ]),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_asso=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_me=dict(type='CategoricalCrossEntropyLoss', loss_weight=1.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))))
info_root = 'data/infos/'
train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(
        type='CollectUnified3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 'timestamp'
        ])
]
test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='CollectUnified3D', keys=['img', 'timestamp'])
]
optimizer = dict(
    type='AdamW',
    lr=1e-05,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=105, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 5
find_unused_parameters = True
runner = dict(type='EpochBasedRunner', max_epochs=5)
fp16 = dict(loss_scale=32.0)
gpu_ids = range(0, 8)
