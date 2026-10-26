# Copyright (c) 2024-2025, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia License.
# To view a copy of this license, visit
# https://github.com/NVlabs/OmniDrive/blob/main/LICENSE
#
# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from ...datasets.utils.constants import IGNORE_INDEX
from mmdet3d.models import builder
from ..dense_heads.llava_llama import LlavaLlamaForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from ..utils.misc import load_model
from ..utils.positional_encoding import pos2posemb2d
import torch.nn as nn
import os
import json
import mmcv
from projects.mmdet3d_plugin.models.utils.misc import MLN
from mmdet.models.utils.transformer import inverse_sigmoid
import time
@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""
    def __init__(self,
                 save_path='./results_vlm/',
                 use_grid_mask=False,
                 embed_dims=256,
                 LID=True,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 depth_num=64,
                 depth_start = 1,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 map_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 lm_head=None,
                 tokenizer=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 frozen=True,
                 use_lora=False,
                 pretrained=None,
                 export_onnx=False,
                 m1_enable=False,
                 m1_execution_mode='observe_only',
                 m1_config=None,
                 m2_enable=False,
                 m2_config=None,
                 m3_enable=False,
                 m3_config=None,
                 worldstream_require_m123=False,
                 ):
        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.save_path = save_path
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.query_pos = nn.Sequential(
            nn.Linear(396, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
        )

        self.ego_pose_pe = MLN(156, export_onnx=export_onnx)

        self.pts_bbox_head.query_pos = self.query_pos
        self.pts_bbox_head.time_embedding = self.time_embedding
        self.pts_bbox_head.ego_pose_pe = self.ego_pose_pe

        if map_head is not None:
            self.map_head = builder.build_head(map_head)
            self.map_head.query_pos = self.query_pos
            self.map_head.time_embedding = self.time_embedding
            self.map_head.ego_pose_pe = self.ego_pose_pe

        if tokenizer is not None:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                        model_max_length=2048,
                                        padding_side="right",
                                        use_fast=False,
                                        )
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer = None
        
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if LID:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - depth_start) / (depth_num * (1 + depth_num))
            coords_d = depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            bin_size = (self.position_range[3] - depth_start) / depth_num
            coords_d = depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self.position_encoder = nn.Sequential(
                nn.Linear(depth_num*3, embed_dims*4),
                nn.ReLU(),
                nn.Linear(embed_dims*4, embed_dims),
            )
        
        if lm_head is not None:
            self.lm_head = load_model(lm_head, use_lora, frozen)

        self.test_flag = False
        self.worldstream_require_m123 = worldstream_require_m123
        self.m3_runtime = None
        if m3_enable:
            from worldstream.m3 import M3Runtime
            self.m3_runtime = M3Runtime(m3_config)
            self._register_load_state_dict_pre_hook(lambda *args: self.m3_runtime.reset())
        self.m2_runtime = None
        self._m2_context = None
        if m2_enable:
            from worldstream.m2 import IncrementalVisionRuntime
            self.m2_runtime = IncrementalVisionRuntime(self.img_backbone, m2_config)
            self._register_load_state_dict_pre_hook(lambda *args: self.m2_runtime.reset())
        self.m1_manager = None
        if m1_enable:
            from worldstream.m1 import M1Config, WorldStateManager
            from dataclasses import replace
            settings = replace(M1Config.load(m1_config), enabled=True,
                               execution_mode=m1_execution_mode)
            self.m1_manager = WorldStateManager(settings)
            # Loading a different checkpoint invalidates external state, without touching Head memory.
            self._register_load_state_dict_pre_hook(lambda *args: self.m1_manager.reset())

    @property
    def with_map_head(self):
        """bool: Whether the detector has a map head."""
        return hasattr(self,
                       'map_head') and self.map_head is not None
        
    
    @property
    def with_lm_head(self):
        """bool: Whether the detector has a lm head."""
        return hasattr(self,
                       'lm_head') and self.lm_head is not None
        

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            if getattr(self, 'm2_runtime', None) is not None and not self.training:
                trace = getattr(self, 'worldstream_execution', None)
                if trace is not None:
                    trace['m2']['runtime_entered'] = True
                if hasattr(self.m2_runtime, 'forward_from_m1'):
                    img_feats, self.m2_diagnostics = self.m2_runtime.forward_from_m1(
                        img, self.m1_manager, self._m2_context)
                else:
                    # Preserve the existing injected-runtime protocol.
                    img_feats, self.m2_diagnostics = self.m2_runtime.forward(
                        img, self.m1_manager.last_result if self.m1_manager else None,
                        self.m1_manager, self._m2_context)
            else:
                img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()

        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B), C, H, W)


        return img_feats_reshaped


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def position_embeding(self, data, memory_centers, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)

        return coords_position_embeding



    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          input_ids, 
                          vlm_labels, 
                          vlm_attn_mask,
                          lane_pts,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        B = data['img'].shape[0]
        location = self.prepare_location(img_metas, **data)

        outs_roi = self.forward_roi_head(location, **data)

        pos_embed = self.position_embeding(data, location, img_metas)
        losses = dict()
        if self.with_pts_bbox:
            outs, det_query = self.pts_bbox_head(img_metas, pos_embed, **data)
            vision_embeded_obj = det_query.clone()

            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses.update(self.pts_bbox_head.loss(*loss_inputs))
            
        if self.with_map_head:
            outs, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            loss_inputs = [lane_pts, outs, img_metas]
            losses.update(self.map_head.loss(*loss_inputs))
        

            
        if self.with_lm_head:
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)
            vlm_loss = self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask, labels=vlm_labels, images=vision_embeded, use_cache=False)
            losses.update(vlm_loss=vlm_loss[0])
            
        if self.with_img_roi_head:
            loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
            losses2d = self.img_roi_head.loss(*loss2d_inputs)
            losses.update(losses2d) 

        return losses

    # @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      input_ids=None,
                      vlm_labels=None,
                      lane_pts=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            
            vlm_labels = torch.nn.utils.rnn.pad_sequence(vlm_labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            vlm_labels = vlm_labels[:, :self.tokenizer.model_max_length]
            vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            input_ids = None
            vlm_labels = None
            vlm_attn_mask = None

        data['img_feats'] = self.extract_feat(data['img'])

        losses = self.forward_pts_train(gt_bboxes_3d,
                                    gt_labels_3d, gt_bboxes,
                                    gt_labels, img_metas, centers2d, 
                                    depths, input_ids, vlm_labels, vlm_attn_mask, lane_pts, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        if not self.test_flag: #for interval evaluation
            if getattr(self, 'm2_runtime', None) is not None:
                self.m2_runtime.reset()
            if self.m1_manager is not None:
                self.m1_manager.reset()
            if self.with_pts_bbox:
                self.pts_bbox_head.reset_memory()
            if self.with_map_head:
                self.map_head.reset_memory()
            self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key not in ['img', 'input_ids']:
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, _m1_observer=None, **data):
        """Test function of point cloud branch."""
        #check
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        pos_embed = self.position_embeding(data, location, img_metas)
        bbox_results = []
        if self.with_pts_bbox:
            outs, det_query = self.pts_bbox_head(img_metas, pos_embed, **data)
            vision_embeded_obj = det_query.clone()
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas)
            for bboxes, scores, labels in bbox_list:
                bbox_results.append(bbox3d2result(bboxes, scores, labels))
        
        lane_results = []
        if self.with_map_head:
            outs, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            lane_results = self.map_head.get_bboxes(outs, img_metas)

        if _m1_observer is not None:
            observed_features = {'backbone': data['img_feats']}
            if self.with_pts_bbox:
                observed_features['object_carrier'] = det_query
            if self.with_map_head:
                observed_features['map_carrier'] = map_query
            _m1_observer(bbox_results, lane_results, observed_features)

        generated_text = []
        m3 = getattr(self, 'm3_runtime', None) if not getattr(self, 'training', False) else None
        if self.with_lm_head and (m3 is not None or not os.path.exists(self.save_path+img_metas[0]['sample_idx'])):
            mmcv.mkdir_or_exist(self.save_path)
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)
            m3_answers = None
            if m3 is not None:
                from worldstream.m3 import CarrierLayout
                trace = getattr(self, 'worldstream_execution', None)
                if trace is not None:
                    trace['m3']['runtime_entered'] = True
                    trace['stage'] = 'm3_generation'
                meta = img_metas[0]
                timestamp = float(data['timestamp'].reshape(-1)[0])
                m3_answers, self.m3_diagnostics = m3.generate(
                    lm_head=self.lm_head, input_ids_list=data['input_ids'][0],
                    vision_carriers=vision_embeded,
                    verified_state=getattr(self, 'm3_verified_state', None),
                    tokenizer=self.tokenizer,
                    m2_diagnostics=getattr(self,'m2_diagnostics',None),
                    context=dict(scene_id=str(meta['scene_token']), frame_id=str(meta['sample_idx']),
                        weights_sha256=getattr(self,'worldstream_audit_identity',{}).get('checkpoint_sha256'),
                        state_epoch=getattr(getattr(self.m1_manager,'store',None),'generation',None),
                        state_version=getattr(getattr(self,'m3_verified_state',None),'state_version',None),
                        state_execution_signature=getattr(getattr(self.m1_manager,'store',None),'execution_signature',None),
                        profile=getattr(self, 'worldstream_profile', False),
                        timestamp=timestamp, camera_ids=list(meta.get('cam_infos', {})) or
                        [os.path.basename(os.path.dirname(f)) for f in meta['filename']]),
                    questions=meta['vlm_labels'].data,
                    layout=CarrierLayout(self.pts_bbox_head.num_extra,
                        det_query.shape[1] - self.pts_bbox_head.num_extra, map_query.shape[1]),
                    do_sample=True, temperature=0.1, top_p=0.75, num_beams=1,
                    max_new_tokens=320, use_cache=True)
                if m3_answers is None or len(m3_answers) != len(data['input_ids'][0]):
                    raise RuntimeError('M3 must return one answer per question')
            for i, input_ids in enumerate(data['input_ids'][0]):
                input_ids = input_ids.unsqueeze(0)
                output_ids = m3_answers[i] if m3 is not None else self.lm_head.generate(
                    inputs=input_ids,
                    images=vision_embeded,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.75,
                    num_beams=1,
                    max_new_tokens=320,
                    use_cache=True
                )
                generated_text.append(
                    dict(
                    Q=img_metas[0]['vlm_labels'].data[i],
                    A=self.tokenizer.batch_decode(output_ids, skip_special_tokens=True),
                    ))
            with open(self.save_path+img_metas[0]['sample_idx'], 'w') as file:
                json.dump(generated_text, file)
        return bbox_results, generated_text, lane_results
    
    def simple_test(self, img_metas, **data):
        """Run state precheck, vision/tasks, state commit, then semantic consumption."""
        request_start = time.perf_counter()
        required = getattr(self, 'worldstream_require_m123', False)
        trace = dict(scene_id=str(img_metas[0]['scene_token']),
            frame_id=str(img_metas[0]['sample_idx']),
            timestamp=float(data['timestamp'].reshape(-1)[0]), completed=False, stage='input_contract',
            identity=getattr(self, 'worldstream_audit_identity', dict(status='unbound_by_launcher')),
            m1=dict(precheck_entered=False, precheck_completed=False,
                    commit_entered=False, commit_completed=False, transaction_status='not_started'),
            m2=dict(runtime_entered=False, execution_completed=False),
            m3=dict(runtime_entered=False, execution_completed=False, requests=[]))
        self.worldstream_execution = trace
        # Per-request diagnostics must never inherit a previous request's success.
        self.m2_diagnostics = self.m3_diagnostics = None
        for name in ('m2_runtime', 'm3_runtime'):
            runtime = getattr(self, name, None)
            if runtime is not None and hasattr(runtime, 'last_diagnostics'):
                runtime.last_diagnostics = {}
        precheck = None
        run_m1 = self.m1_manager is not None
        try:
            if required:
                if (getattr(self, 'training', False) or not run_m1
                        or not self.m1_manager.config.enabled
                        or getattr(self, 'm2_runtime', None) is None
                        or getattr(self, 'm3_runtime', None) is None
                        or not self.with_pts_bbox or not self.with_map_head or not self.with_lm_head):
                    raise RuntimeError('Formal WorldStream inference requires M1, M2, M3 and native task heads')
                if len(data.get('input_ids', ())) != 1 or not len(data['input_ids'][0]):
                    raise ValueError('Formal WorldStream VLM request requires at least one language prompt')
            m1_observer = None
            if getattr(self, 'm2_runtime', None) is not None:
                meta = img_metas[0]
                self._m2_context = dict(scene_id=str(meta['scene_token']),
                    frame_id=str(meta['sample_idx']),
                    camera_ids=list(meta.get('cam_infos', {})) or
                        [os.path.basename(os.path.dirname(f)) for f in meta['filename']],
                    coordinate_space='preprocessed_image')
                self._m2_context['timestamp'] = float(data['timestamp'].reshape(-1)[0])
            self.m3_verified_state = None
            run_m1 = self.m1_manager is not None
            if run_m1:
                from worldstream.m1.omnidrive_adapter import frame_from_model, outputs_from_model, append_result, measure_transfers
                import hashlib
                settings = self.m1_manager.config
                adapter_start = time.perf_counter()
                with measure_transfers() as input_transfers:
                    frame = frame_from_model(img_metas, data, settings,
                                             getattr(self, 'm1_source_context', None))
                parameter_signature=[(name,tuple(p.shape),str(p.dtype),str(p.device),p._version) for name,p in self.named_parameters()]
                execution_policy=(getattr(self,'training',False),torch.is_autocast_enabled(),
                    torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                    torch.get_autocast_gpu_dtype(),torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                    torch.are_deterministic_algorithms_enabled())
                suffix=repr(parameter_signature)+repr(execution_policy)
                frame.execution_signature = hashlib.sha256((frame.execution_signature+suffix).encode()).hexdigest()
                adapter_ms = (time.perf_counter() - adapter_start) * 1000
                trace['m1']['precheck_entered'] = True
                trace['stage'] = 'm1_precheck'
                precheck=self.m1_manager.precheck_with_observation(frame, True)
                trace['m1'].update(precheck_completed=True,
                    state_epoch=precheck.diagnostics.get('state_epoch'),
                    valid_states=len(precheck.valid_anchor_ids),
                    expired_states=len(precheck.diagnostics.get('expired_anchor_ids', [])),
                    historical_candidates=len(precheck.association_candidate_ids),
                    transaction_status=precheck.diagnostics.get('transaction_status'),
                    full_refresh_reasons=precheck.diagnostics.get('full_refresh_reasons', []))
                def publish_precheck():
                    trace['stage'] = 'history_snapshot'
                    if getattr(self,'m3_runtime',None) is not None:
                        from worldstream.m3.world_state import freeze_verified_state
                        self.m3_verified_state=freeze_verified_state(self.m1_manager,frame)
                    if getattr(self,'m2_runtime',None) is not None:self._m2_context.update(execution_signature=frame.execution_signature)
                    if getattr(getattr(self,'m2_runtime',None),'config',None) is not None:
                        self._m2_context['state_epoch'] = precheck.diagnostics.get('state_epoch')
                    precheck.diagnostics['timings_ms']['input_adapter_including_transfers']=adapter_ms
                    precheck.diagnostics['timings_ms']['input_gpu_to_cpu_transfer']=input_transfers['gpu_to_cpu_transfer_ms']
                    precheck.diagnostics['input_gpu_to_cpu_transfer_bytes']=input_transfers.get('gpu_to_cpu_transfer_bytes')
                publish_precheck()

                def m1_observer(boxes, lanes, features):
                    trace['stage'] = 'm1_output_adapter'
                    output_start = time.perf_counter()
                    with measure_transfers() as output_transfers:
                        outputs = outputs_from_model(frame, boxes, lanes, features, settings)
                    if getattr(self, 'm2_runtime', None) is not None and not self.with_img_neck:
                        outputs.execution_receipt = self.m2_diagnostics
                    output_ms = (time.perf_counter() - output_start) * 1000
                    trace['m1']['commit_entered'] = True
                    trace['stage'] = 'm1_commit'
                    self.m1_manager.commit(frame, outputs, precheck)
                    trace['m1'].update(commit_completed=True,
                        transaction_status=precheck.diagnostics.get('transaction_status'))
                    self.m1_manager.last_commit_diagnostics['output_adapter_including_transfers_ms'] = output_ms
                    self.m1_manager.last_commit_diagnostics['output_gpu_to_cpu_transfer_ms'] = output_transfers['gpu_to_cpu_transfer_ms']
                    self.m1_manager.last_commit_diagnostics['output_gpu_to_cpu_transfer_bytes'] = output_transfers.get('gpu_to_cpu_transfer_bytes')
                    self.m1_manager.last_commit_diagnostics['m1_total_including_adapters_ms'] = (
                        adapter_ms + output_ms + precheck.diagnostics['timings_ms']['precheck_total']
                        + self.m1_manager.last_commit_diagnostics['commit_total_ms']
                        + self.m1_manager.last_commit_diagnostics.get('computation_source_metadata_ms', 0.))
            trace['stage'] = 'm2_vision'
            data['img_feats'] = self.extract_img_feat(data['img'])
            trace['stage'] = 'detection_map'
            bbox_list = [dict() for i in range(len(img_metas))]
            if m1_observer is None:
                bbox_pts, generated_text, lane_results = self.simple_test_pts(img_metas, **data)
            else:
                bbox_pts, generated_text, lane_results = self.simple_test_pts(
                    img_metas, _m1_observer=m1_observer, **data)
            if required:
                trace['stage'] = 'execution_contract'
                m2 = self.m2_diagnostics or {}
                m3 = self.m3_diagnostics or {}
                if (not trace['m1']['precheck_completed'] or not trace['m1']['commit_completed']
                        or trace['m1']['transaction_status'] not in ('committed', 'rejected')
                        or not trace['m2']['runtime_entered'] or not m2.get('execution_completed')
                        or m2.get('executor') != 'WindowExecutor'
                        or not trace['m3']['runtime_entered'] or not m3.get('execution_completed')
                        or len(m3.get('requests', [])) != len(data['input_ids'][0])
                        or any(r.get('executor') != 'PrefillExecutor' or not r.get('prefill_completed')
                               or not r.get('decode_completed') for r in m3.get('requests', []))):
                    raise RuntimeError('Formal WorldStream request bypassed or did not complete M1/M2/M3')
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
            bbox_list[0]['text_out'] = generated_text
            bbox_list[0]['lane_results'] = lane_results
            trace['completed'] = True
            trace['stage'] = 'completed'
            return bbox_list
        except Exception as error:
            trace['error'] = dict(type=type(error).__name__, message=str(error), stage=trace['stage'])
            if run_m1:
                self.m1_manager.abort_pending()
            raise
        finally:
            # Host elapsed only: no CUDA synchronization in ordinary inference.
            trace['host_elapsed_ms'] = (time.perf_counter() - request_start) * 1000
            trace['timing_method'] = 'host wall; GPU completion synchronized only by profiling launcher'
            if precheck is not None:
                trace['m1']['transaction_status'] = precheck.diagnostics.get('transaction_status')
                trace['m1']['timings_ms'] = dict(precheck.diagnostics.get('timings_ms', {}))
                if trace['m1']['commit_completed']:
                    trace['m1']['commit_timings_ms'] = {k: v for k, v in
                        self.m1_manager.last_commit_diagnostics.items() if k.endswith('_ms')}
            # Reuse the existing runtime diagnostics and audit/benchmark sinks.
            for name in ('m2', 'm3'):
                if not trace[name]['runtime_entered']:
                    continue
                diag = getattr(self, name + '_diagnostics', None)
                if diag is None:
                    diag = getattr(getattr(self, name + '_runtime', None), 'last_diagnostics', {})
                keys = ('execution_completed', 'actual_window_reuse', 'actual_skipped_windows',
                        'full_recompute', 'full_recompute_reason', 'cache_publication_error', 'executor') if name == 'm2' else ('execution_completed',)
                trace[name].update({k: diag[k] for k in keys if k in diag})
                if name == 'm3':
                    trace[name]['requests'] = [dict(
                        request_index=r.get('request_index'),
                        executor=r.get('executor'), matched_prefix=r.get('matched_prefix'),
                        historical_text_consumed=r.get('historical_text_consumed', False),
                        history_reason=r.get('state_retrieval', {}).get('reason'),
                        reused_tokens=r.get('reused_tokens', 0), full_prefill=r.get('full_prefill', False),
                        full_prefill_reason=r.get('full_prefill_reason'),
                        prefill_completed=r.get('prefill_completed', False),
                        decode_completed=r.get('decode_completed', False)) for r in diag.get('requests', [])]
                    for compact,detail in zip(trace[name]['requests'],diag.get('requests',[])):
                        for key in ('reuse_strategy','task_type','task_profile','layout_signature','layout_permutation',
                            'stable_groups','changed_groups','mandatory_refresh_groups','m1_state_valid',
                            'm2_mapping_available','matched_prefix_tokens','approved_prefix_tokens','effective_prefill_tokens',
                            'prefix_rejection_reason','cache_invalidation_reason','layout_fallback_reason',
                            'layout_preparation_ms','prefix_matching_ms','cache_branch_ms','prefill_ms','decode_ms','end_to_end_ms'):
                            if key in detail:compact[key]=list(detail[key]) if isinstance(detail[key],tuple) else detail[key]
            if precheck is not None and settings.logging and settings.log_path:
                # Commit precedes M3; serialize once after M3 to include the entire request.
                commit_diag = self.m1_manager.last_commit_diagnostics if trace['m1']['commit_completed'] else {}
                append_result(settings.log_path, precheck, commit_diag,
                    self.m1_manager.export_state_audit(), experiment=dict(module_execution=trace))

