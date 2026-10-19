# M1 pre-change root cause audit

HEAD: 15a3e840bf41726bcd538cc09456664300ec29c8.

Identity archived Development population: 5798. Full multilabel and mutually exclusive primary breakdowns are in identity_rejection_diagnostics.json. The old producer exits before RGB measurement for displacement >0.5m and any legal competitor. It cannot establish how many of these have insufficient RGB evidence; new diagnostic probes must measure them without treating proposals as qualifications. Historical sources older than the immediately previous committed image cannot use that image's tracks. Static epipolar geometry also censors motion even though identity does not imply stationarity.

Surface path applies finite flow, FB, photo/2D texture, calibrated rays and complete-neighbourhood erosion, then requires **every pixel** to pass adjacent .5 degree parallax AND independently measured three-view loop/ray closure. Missing third history is all Unknown. Dynamic swept/occlusion masks can remove support afterwards; periodic refresh removes effective area without changing observation area. Full sequential and marginal pixel gates will be measured in a pre-change probe, not inferred from the historical final area.

Old Local R12 has 169 false accepted points: 46 measured low-parallax and 123 Unknown Root Cause. The original RGB, flow, pose and independent annotation/LiDAR traces remain read-only and are SHA256 pinned in surface_error_taxonomy.json. No ownership attribution is invented.

Exposure audit: 201 scenes, 68 logs. Available unexposed endpoint logs: 0. Calibration/Heldout/Final use only remaining logs and remain empty if exhausted. All train-derived selections disclose original-model training exposure. At least 59 fully scored independent zero-error accepted logs are needed for the physical risk bound; missing units are never filled by repeated frames or objects.

The experiment protocol, old source manifest, exposure inventory and data selection were frozen before implementation. Baseline regression: 420 passed, one CUDA-only skipped under sandbox; host GPU read-only check reports idle. Baseline quality tables remain in the historical report, to be distinguished from newly executed development replay.

## Measured uncensored baseline probe

{
  "population": {
    "objects": 5798,
    "scenes": 20,
    "correct": 2655,
    "wrong": 20,
    "unscorable": 3123
  },
  "outside_half_metre": {
    "objects": 645,
    "scenes": 19,
    "correct": 219,
    "wrong": 11,
    "unscorable": 415
  },
  "observable_outside_half_metre": {
    "objects": 1,
    "scenes": 1,
    "correct": 1,
    "wrong": 0,
    "unscorable": 0
  },
  "observable_with_competition": {
    "objects": 196,
    "scenes": 5,
    "correct": 76,
    "wrong": 0,
    "unscorable": 120
  },
  "measured_reasons": {
    "source_time_mismatch": {
      "objects": 107,
      "scenes": 16,
      "correct": 13,
      "wrong": 2,
      "unscorable": 92
    },
    "RGB_tracks_missing": {
      "objects": 3560,
      "scenes": 20,
      "correct": 1470,
      "wrong": 16,
      "unscorable": 2074
    },
    "crop_track_fraction": {
      "objects": 3919,
      "scenes": 20,
      "correct": 1721,
      "wrong": 13,
      "unscorable": 2185
    },
    "spatial_quadrants": {
      "objects": 3699,
      "scenes": 20,
      "correct": 1550,
      "wrong": 16,
      "unscorable": 2133
    },
    "appearance_descriptor": {
      "objects": 4274,
      "scenes": 20,
      "correct": 1840,
      "wrong": 18,
      "unscorable": 2416
    }
  },
  "no_common_view": {
    "objects": 1037,
    "scenes": 20,
    "correct": 519,
    "wrong": 2,
    "unscorable": 516
  },
  "note": "Observable crop tracks are feasibility evidence, not foreground ownership or true identity"
}

## Measured surface funnel

{
  "valid_flow": {
    "total_pixels": 304128000,
    "standalone_population": 304128000,
    "alone_rejected": 20498384,
    "incremental_rejected": 20498384,
    "retained": 283629616
  },
  "fb_consistency": {
    "total_pixels": 304128000,
    "standalone_population": 304128000,
    "alone_rejected": 236965557,
    "incremental_rejected": 218159356,
    "retained": 65470260
  },
  "photo_and_texture": {
    "total_pixels": 304128000,
    "standalone_population": 304128000,
    "alone_rejected": 183321572,
    "incremental_rejected": 16379126,
    "retained": 49091134
  },
  "ray_geometry": {
    "total_pixels": 304128000,
    "standalone_population": 304128000,
    "alone_rejected": 234787968,
    "incremental_rejected": 13027094,
    "retained": 36064040
  },
  "complete_neighbourhood": {
    "total_pixels": 304128000,
    "standalone_population": 304128000,
    "alone_rejected": 282011181,
    "incremental_rejected": 13947221,
    "retained": 22116819
  },
  "adjacent_observability": {
    "total_pixels": 304128000,
    "standalone_population": 22116819,
    "alone_rejected": 21587096,
    "incremental_rejected": 21587096,
    "retained": 529723
  },
  "long_baseline_match": {
    "total_pixels": 304128000,
    "standalone_population": 22116819,
    "alone_rejected": 21683200,
    "incremental_rejected": 337666,
    "retained": 192057
  },
  "trajectory_loop": {
    "total_pixels": 304128000,
    "standalone_population": 22116819,
    "alone_rejected": 12331285,
    "incremental_rejected": 26883,
    "retained": 165174
  },
  "third_ray_reprojection": {
    "total_pixels": 304128000,
    "standalone_population": 22116819,
    "alone_rejected": 19804645,
    "incremental_rejected": 156138,
    "retained": 9036
  },
  "dynamic_occlusion_and_changed": {
    "retained": 5823
  },
  "effective_after_refresh": {
    "retained": 5529
  }
}
