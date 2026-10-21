from types import SimpleNamespace
import numpy as np
import pytest
from worldstream.m1.dqtrack_export import export_frame


def info():
    return dict(token='sample-token',scene_token='scene',timestamp=100000000,
        lidar2ego_rotation=[1,0,0,0],lidar2ego_translation=[0,0,0],
        ego2global_rotation=[1,0,0,0],ego2global_translation=[0,0,0])


def result(centers):
    centers=np.asarray(centers,dtype=float);n=len(centers)
    tensor=np.zeros((n,9));tensor[:,7]=1
    return dict(eval_det=False,boxes_3d=SimpleNamespace(gravity_center=centers,
        dims=np.tile([4,2,2],(n,1)),yaw=np.zeros(n),tensor=tensor),
        track_ids=np.arange(n)+41,track_scores=np.arange(n)*.1+.7,
        labels_3d=np.zeros(n,dtype=int),scores_3d=np.zeros(n))


def test_range_filter_keeps_original_id_score_and_index():
    raw=result([[100,0,1],[2,0,1]])
    rows,d=export_frame(raw,info(),['car'],{'car':50})
    assert d['range_filtered']==1 and d['exported']==1
    assert rows[0]['tracking_id']=='42'
    assert rows[0]['tracking_score']==pytest.approx(.8)
    assert rows[0]['size']==[2,4,2]
    assert rows[0]['translation']==[2,0,1]


def test_lidar_ego_world_rotation_and_velocity():
    i=info();i['lidar2ego_translation']=[1,0,0]
    i['ego2global_rotation']=[2**-.5,0,0,2**-.5];i['ego2global_translation']=[10,20,0]
    rows,d=export_frame(result([[2,0,1]]),i,['car'],{'car':50})
    assert np.allclose(rows[0]['translation'],[10,23,1])
    assert np.allclose(rows[0]['velocity'],[0,1])
    assert np.allclose(rows[0]['rotation'],i['ego2global_rotation'])


@pytest.mark.parametrize('field,value',[('track_ids',[-1]),('track_ids',[1.2]),
    ('track_scores',[float('nan')]),('labels_3d',[7])])
def test_illegal_rows_are_rejected(field,value):
    r=result([[0,0,1]]);r[field]=np.array(value)
    rows,d=export_frame(r,info(),['car'],{'car':50})
    assert rows==[] and d['rejected'][0]['source_index']==0


def test_duplicates_and_shape_mismatch():
    r=result([[0,0,1],[1,0,1]]);r['track_ids'][:]=2
    rows,d=export_frame(r,info(),['car'],{'car':50})
    assert rows==[] and len(d['rejected'])==2
    r['track_ids']=np.array([1])
    with pytest.raises(ValueError):export_frame(r,info(),['car'],{'car':50})


def test_empty_det_only_and_convention():
    assert export_frame(None,info(),['car'],{'car':50})[0]==[]
    r=result([[0,0,1]]);r['eval_det']=True
    with pytest.raises(ValueError):export_frame(r,info(),['car'],{'car':50})
    with pytest.raises(ValueError):export_frame(None,info(),['car'],{'car':50},box_convention='unknown')


def test_invalid_pose_rejects_frame():
    i=info();i['ego2global_rotation']=[0,0,0,0]
    with pytest.raises(ValueError):export_frame(result([[0,0,1]]),i,['car'],{'car':50})


def test_restricted_checkpoint_loader_accepts_float_metadata(tmp_path):
    import torch
    from collections import OrderedDict
    from tools.worldstream_m1.run_dqtrack_source import load_tensor_checkpoint
    path=tmp_path/'tensor.pth'
    torch.save(dict(state_dict=OrderedDict(weight=torch.ones(2)),meta={'score':.37}),path)
    loaded=load_tensor_checkpoint(path)
    assert loaded['meta']['score']==.37
    assert torch.equal(loaded['state_dict']['weight'],torch.ones(2))


def test_restricted_checkpoint_rejects_executable_global(tmp_path):
    import torch,os,pickle
    from tools.worldstream_m1.run_dqtrack_source import load_tensor_checkpoint
    class Executable:
        def __reduce__(self):return os.system,('touch '+str(tmp_path/'must_not_exist'),)
    path=tmp_path/'untrusted.pth';torch.save({'bad':Executable()},path)
    with pytest.raises(pickle.UnpicklingError):load_tensor_checkpoint(path)
    assert not (tmp_path/'must_not_exist').exists()


def test_gpu_busy_never_starts_work(monkeypatch):
    import tools.worldstream_m1.run_dqtrack_source as runner
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'12345\n')
    with pytest.raises(TimeoutError):runner.wait_for_gpu(0)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'')
    runner.wait_for_gpu(0)


def test_exported_rows_align_with_existing_m1(tmp_path,config,make_frame,make_prediction):
    import json
    from dataclasses import replace
    from tools.worldstream_m1.prepare_dqtrack_results import prepare
    from worldstream.m1.dqtrack_bridge import align
    i=info();rows,_=export_frame(result([[0,0,10]]),i,['car'],{'car':50})
    sample=dict(token=i['token'],scene_token=i['scene_token'],timestamp=i['timestamp'])
    data=prepare({'results':{i['token']:rows}},[sample],'source-run','source-config','a'*64)
    path=tmp_path/'normalized.json';path.write_text(json.dumps(data))
    frame=make_frame(frame_id=i['token'])
    hints,d=align(frame,[make_prediction(semantic='car')],replace(config,dqtrack_results_path=str(path)))
    assert d['aligned']==1 and hints[0]['tracking_id']=='41'
    assert not hints[0]['certified']
