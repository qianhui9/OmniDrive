"""Offline approximate input probe. Never imported by the serving runtime.

Only decoded current predictions and current ego telemetry enter this module.
All predictions meeting the existing M1 0.3 score contract are serialized.
Excluded low-score counts are explicit; this is a lossy approximate input,
never a claim that decoded heads exhaust the physical world.
"""
import json

import numpy as np
import torch

IMAGE_TOKEN = -200
MODES = ('full', 'current_compact', 'compact_full')


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite current perception')
    return array


def detailed_current_context(boxes, lanes, ego):
    from worldstream.m1.omnidrive_adapter import CLASS_NAMES
    objects = []
    for batch in boxes:
        centers = _array(batch['boxes_3d'].gravity_center)
        dims = _array(batch['boxes_3d'].dims)
        values = _array(batch['boxes_3d'].tensor)
        labels = _array(batch['labels_3d']).astype(int)
        scores = _array(batch['scores_3d'])
        for center, size, val, label, score in zip(centers, dims, values, labels, scores):
            if score < .3: continue
            objects.append([CLASS_NAMES[label], *np.round(center, 1).tolist(),
                            *np.round(size, 1).tolist(), round(float(val[6]), 2),
                            *np.round(val[7:9], 1).tolist(), round(float(score), 2)])
    maps = []
    for points, scores in lanes:
        points, scores = _array(points), _array(scores)
        keep = scores.max(axis=-1) >= .3 if scores.ndim > 1 else scores >= .3
        maps.append(dict(control_points=np.round(points[keep], 1).tolist(),
                         scores=np.round(scores[keep], 2).tolist(),
                         excluded_low_score=int((~keep).sum())))
    payload = dict(objects=objects, lane_predictions=maps,
                   excluded_low_score_objects=sum(int((_array(b['scores_3d']) < .3).sum()) for b in boxes),
                   ego_telemetry=np.round(_array(ego), 2).reshape(-1).tolist())
    return ('Current model predictions in ego coordinates (x forward, y left), metres. '
            'Objects=[class,x,y,z,width,length,height,yaw,vx,vy,score]. '
            'Lane control points have no certified topology or markings. '
            'Missing objects/attributes remain unknown.\n' +
            json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')


def current_context(boxes, lanes, ego):
    """Fixed lossy semantic format, frozen before any independent outcomes.

    Every >=0.3 object retains class and xy. Every >=0.3 predicted lane retains
    first/middle/last xy control points. Integer-metre quantization deliberately
    discards size, z, yaw, speed, score and finer map shape. All such information
    loss is tested against native Planning, never assumed task-sufficient.
    """
    objects=[];maps=[];excluded=0
    for batch in boxes:
        centers=_array(batch['boxes_3d'].gravity_center)
        labels=_array(batch['labels_3d']).astype(int);scores=_array(batch['scores_3d'])
        for center,label,score in zip(centers,labels,scores):
            if score < .3:excluded+=1;continue
            objects.append(f'{label}:{int(np.rint(center[0]))},{int(np.rint(center[1]))}')
    for points,scores in lanes:
        points,scores=_array(points),_array(scores)
        keep=scores.max(axis=-1)>=.3 if scores.ndim>1 else scores>=.3
        for line in points[keep]:
            xy=np.rint(line[[0,len(line)//2,-1],:2]).astype(int).reshape(-1)
            maps.append(','.join(map(str,xy)))
    return ('Current predictions; integer metres; x forward,y left. '
            'Object codes 0 car,1 truck,2 construction,3 bus,4 trailer,5 barrier,6 motorcycle,7 bicycle,8 pedestrian,9 cone. '
            'O=class:x,y; L=start,mid,end xy lane control points. No topology/signals/appearance.\n'
            'O '+';'.join(objects)+'\nL '+';'.join(maps)+
            '\nEgo telemetry '+','.join(str(round(float(x),1)) for x in _array(ego).reshape(-1))+
            f'\nExcluded objects below native .3 score: {excluded}.\n')


def probe_inputs(input_ids, images, tokenizer, context, mode):
    """Replace the image sentinel, preserving every original question token.

    Full is an object-identity passthrough. B removes the sentinel altogether,
    so negative IDs cannot reach the text embedding. C retains it once. Native
    Llava code creates positions/causal attention from the resulting sequence.
    """
    if mode not in MODES:
        raise ValueError('Unknown offline input probe')
    if mode == 'full':
        return input_ids, images
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError('Probe supports one original question at a time')
    where = (input_ids[0] == IMAGE_TOKEN).nonzero().reshape(-1)
    if len(where) != 1:
        raise ValueError('Exactly one native image insertion point required')
    tokens = tokenizer(context, add_special_tokens=False).input_ids
    added = torch.tensor(tokens, dtype=input_ids.dtype, device=input_ids.device)[None]
    index = int(where[0])
    suffix = input_ids[:, index:] if mode == 'compact_full' else input_ids[:, index+1:]
    return torch.cat((input_ids[:, :index], added, suffix), dim=1), (images if mode == 'compact_full' else None)
