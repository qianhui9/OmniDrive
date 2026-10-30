"""Protect the independent denominators and UNKNOWN handling of the offline audit."""
import pytest
import torch

from tools.worldstream_m123.final_quality_analysis import detection_frame, aggregate_detection, association, archive_receipts
from tools.worldstream_m123.final_quality_capture import feature_error


def anchor(aid, x, cls='car'):
    return dict(anchor_id=aid, anchor_type='object', observed=True, source_frame='f',
                source_timestamp=1., evidence_origin='actual_perception', center_ego=[x,0.,0.],
                semantic_class=cls)


def gt(aid, x):
    return dict(annotation_id=aid, instance_id=aid, semantic_class='car',
                center_ego=[x,0.,0.], visible=True, cameras=['CAM_FRONT'])


def test_unmatched_prediction_is_unknown_and_gt_denominator_includes_missed_object():
    result=aggregate_detection([detection_frame(dict(frame_id='f',timestamp=1.),
        [anchor('a',5.),anchor('unadjudicated',40.)],[gt('g',5.),gt('missed',15.)])])
    assert result['groups']['visible_GT']['TP']==1
    assert result['groups']['visible_GT']['FN']==1
    assert result['groups']['visible_GT']['proven_class_recall']==.5
    assert result['UNKNOWN_predictions']==1
    assert result['FP'] is None
    assert result['unconditional_class_precision_bounds']==[.5,1.]


def test_ambiguous_duplicate_is_unknown_gt_instead_of_proven_false_negative():
    result=detection_frame(dict(frame_id='f',timestamp=1.),
        [anchor('a',5.),anchor('b',5.1)],[gt('g',5.)])
    assert result['groups']['visible_GT']['TP']==0
    assert result['groups']['visible_GT']['FN']==0
    assert result['groups']['visible_GT']['UNKNOWN_GT']==1


def test_class_independent_geometry_match_exposes_wrong_class():
    result=detection_frame(dict(frame_id='f',timestamp=1.),[anchor('a',5.,'truck')],[gt('g',5.)])
    assert result['matched']==1 and result['class_correct']==0
    assert result['confirmed_class_FP']==1
    assert result['groups']['visible_GT']['FN']==1


def test_returned_feature_camera_tiles_and_token_metrics():
    reference=torch.ones(2,3,8,8)
    actual=reference.clone();actual[1,:,4:,4:]+=2.
    result=feature_error(actual,reference)
    assert result['relative_l2']==pytest.approx(2**-.5)
    assert result['per_camera'][0]['max_abs']==0.
    assert result['per_camera'][1]['relative_l2']==pytest.approx(1.)
    assert len(result['tiles_4x4'])==32
    assert sum(t['relative_l2']>0 for t in result['tiles_4x4'])==4
    assert result['spatial_token_relative_l2_quantiles']['0.99']==2.


def test_visible_association_coverage_uses_visible_numerator():
    rows=[];objects={}
    for i in range(2):
        fid=f'f{i}';a=anchor('a',5.);a.update(source_frame=fid,source_timestamp=float(i))
        hidden=gt('hidden',5.);hidden['visible']=False
        objects[fid]=[hidden,gt('visible_missed',15.)]
        rows.append(dict(scene_id='s',frame_id=fid,timestamp=float(i),frame_index=i,
            committed_world=dict(anchors=[a]),visual_quality=dict(association_pairs=[['a',0]] if i else [])))
    result=association(rows,objects)
    assert result['counts']['correct_links']==1
    assert result['counts'].get('correct_visible_links',0)==0
    assert result['counts']['consecutive_visible_GT_opportunities']==1
    assert result['effective_association_coverage']==0.


def test_archive_manifest_survives_reanalysis_from_compressed_only_checkout(tmp_path):
    import json
    stage=tmp_path/'stage';stage.mkdir()
    raw=stage/'requests.jsonl';raw.write_text('{"receipt": 1}\n')
    archive_receipts(tmp_path)
    original=json.loads((tmp_path/'receipt_archive.json').read_text())
    raw.unlink()
    archive_receipts(tmp_path)
    assert json.loads((tmp_path/'receipt_archive.json').read_text())==original
