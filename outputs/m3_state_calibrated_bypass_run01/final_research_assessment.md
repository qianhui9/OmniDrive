# Final research assessment

## Frozen acceptance outcome and limits

Final verdict **NO_GO**. The acceptance rules were frozen before TEST, including no new severe errors. There are 2 newly severe outputs, 1 unique frame/question cases; correlated rounds are not independent cases.

Addressable3Q B→C 1.370768x. Known required-fact accuracy B=64.893617%, C=70.212766%, coverage=94/144, UNKNOWN=50. Conservative all-request B-minus-C loss bounds=[-0.06944444444444445, 0.0]. Severe totals B=24, C=22; reductions in other cases do not cancel newly severe outputs.

- Frame `dcdbd1100ef14eb0b00e272bc01b8b21`, Where is the nearest pedestrian relative to the vehicle?: C='The nearest pedestrian is to the front-right.'; B='The nearest pedestrian is on the sidewalk to the left of the vehicle, walking parallel to the road.'; reference={'known': True, 'task': 'position', 'semantic_class': 'pedestrian', 'positive': True, 'motion': None, 'available_classes': ['pedestrian'], 'directions': ['front', 'left'], 'annotation_ids': ['4088a55bc0de4a02baaab2987023d275', '936eec700a1c45ba929974967ce6d47b', '50cc1cd193544afab9acfad5a4398ac0'], 'selected_annotation': '4088a55bc0de4a02baaab2987023d275', 'source_frame': 'dcdbd1100ef14eb0b00e272bc01b8b21', 'center_ego': [22.76690274220664, 4.059762664219306, 1.221062558902588], 'required_facts': ['target_class', 'longitudinal_direction', 'lateral_direction', 'current_observation'], 'metric': 'required_current_visible_object_facts', 'profile': {'intent': 'position', 'classes': ['pedestrian'], 'directions': [], 'near_m': 50.0, 'direct': False, 'selector': 'nearest', 'required': ['target_class', 'longitudinal_direction', 'lateral_direction', 'current_observation'], 'reason': 'structured_position'}}; path=fast_current_fact. Current facts=[{'anchor_id': '07aed9dae37340a997535ad99138e243:g199:a218', 'semantic_class': 'pedestrian', 'center_ego': [27.5339063406459, -8.744233004216767, 0.8524219018794383], 'directions': ['front', 'right'], 'motion': '', 'velocity_ego': [], 'confidence': 0.4530470371246338, 'cameras': ['CAM_FRONT'], 'source_frame': 'dcdbd1100ef14eb0b00e272bc01b8b21', 'source_timestamp': 1538448847.147828, 'state_version': 7, 'state_epoch': 199, 'evidence_origin': 'actual_perception', 'identity_qualified': False, 'observation_kind': 'current_photometric_support', 'identity_required': False, 'association_unambiguous': False}]; offline cause=unmatched_current_detection_identity_UNKNOWN; reference in current supported candidates=False. An UNKNOWN identity match does not erase a KNOWN opposite-direction fact error.
  Temporal companion: {'Q': 'Where is the nearest pedestrian relative to the vehicle?', 'A': ['The nearest pedestrian is to the front-right.']}; actual skip=True.

Fast evidence sources (formal events; repetitions/workloads are correlated):

- current_photometric_support: events=108, scored=104, UNKNOWN=4, score=0.9615384615384616, unique scored frame/questions=32, scenes=3
- current_detector_center_projection_not_reuse_certificate: events=10, scored=10, UNKNOWN=0, score=0.6, unique scored frame/questions=2, scenes=1

The offline same-detection old/new snapshot comparison has 36 jointly scored candidates, 1 wrong-to-right and 1 right-to-wrong changes; UNKNOWN/missing legacy=4. This does not prove an independent online E2E or net generalization improvement. Resolving the old Car diagnostic is insufficient. TEST outcomes never change the policy.

Temporal3Q veto events={'unnecessary_veto': 8, 'unknown_veto': 2}; unique frame/question cases={'beneficial_veto': 0, 'unnecessary_veto': 4, 'unknown_veto': 1}. Temporal independent scored quality gain=False. Current→Temporal3Q speed ratio=0.954503x; delta sceneCI=[-12.20893335994333, 103.3240512770135]ms. Separate timing phases and three logs limit interpretation.

Natural3Q actual skips=0; no established natural-QA speed benefit. Preset mixed3Q B→C=1.065942x, skips=22/144; the introduced1/3 structured slots are not a natural-distribution estimate.

DEV selects temporal=False, confidence=0.45, risk=0.75. This is the best observed of four predeclared DEV candidates, not a TEST-qualified deployment policy or a global optimum. Default research switches stay off. TEST has 3 independent logs/24 frames; rounds are correlated. UNKNOWNs do not certify complete driving, visual appearance, absence or motion quality. Implementation and experiments are complete; research acceptance is a separate outcome.
