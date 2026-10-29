# Training exposure audit

Local original training configs and the embedded checkpoint configuration are inspected separately from WRSAR training. The original pipeline uses nuScenes train image/Conv/VQA inputs, so physical-train WRSAR scenes cannot be called model-unseen. The local checkpoint metadata reports exp_name=mask_eva_lane_det_vlm.py, iter=10547, epoch=1, time=Fri Apr 18 04:41:08 2025. Its embedded config directly specifies nuscenes2d_ego_temporal_infos_train.pkl and train Conv/VQA. An exact scene-by-scene training execution ledger and authenticated acquisition manifest are unavailable.

- train: 28130 frames / 700 scenes; compatible first8 six-camera linked segments 700; historical used-log scenes 700.
- val: 6019 frames / 150 scenes; compatible first8 six-camera linked segments 150; historical used-log scenes 150.

Unused compatible official validation scenes: 0. Metadata and per-source SHA256 receipts are in dataset_split_provenance.json. Exclusion uses entire logs across all historical M3/M123 plans and records, including diagnostics. Trainval variants are alternate indexes, not new datasets. The official unlabeled test split is not an eligible substitute. No new TEST or GPU baseline is scheduled without an eligible independent pool.

Upstream primary configuration also specifies train metadata and train QA: [NVlabs configuration](https://raw.githubusercontent.com/NVlabs/OmniDrive/main/projects/configs/OmniDrive/mask_eva_lane_det_vlm.py). The supplied GitHub fork page was unavailable through browsing; local HEAD is the execution authority.
