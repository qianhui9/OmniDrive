"""R0-required utilities; historical alternative runners have been retired."""

import json

from pathlib import Path

from .benchmark.common import write

QUESTIONS=[
    'What type of vehicle is directly ahead?',
    'Is there a car in front?',
    'Where is the nearest pedestrian relative to the vehicle?',
]


def append(path,value):
    with Path(path).open('a') as handle:handle.write(json.dumps(value,allow_nan=False)+'\n')


def encode_questions(model,questions,location):
    """Exactly the deployed Llava conversation prompt, not bare question IDs."""
    from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import preprocess,DEFAULT_IMAGE_TOKEN
    # NuScenesDataset.get_data_info publishes only the city prefix to the
    # loader; raw info.location also contains the site name (e.g. seaport).
    location=location.split('-')[0]
    sources=[[dict(**{'from':'human'},value=DEFAULT_IMAGE_TOKEN+'\n'+f'You are driving in {location}. '+q),
              dict(**{'from':'gpt'},value='')] for q in questions]
    ids=preprocess(sources,model.tokenizer,True,False)['input_ids']
    device=next(model.lm_head.parameters()).device
    return [v.to(device) for v in ids]


def validate_native_prompt_context(model,dataset,index,scenes,out):
    """Fail before any timed inference if reconstructed native inputs differ."""
    import torch
    from mmcv.parallel import collate,scatter
    checked=[]
    for scene in scenes:
        token=scene['tokens'][0];i=index[token]
        probe=scatter(collate([dataset[i]],samples_per_gpu=1),[0])[0]
        labels=list(probe['img_metas'][0][0]['vlm_labels'].data)[1:4]
        native=probe['input_ids'][0][0][1:4]
        rebuilt=encode_questions(model,labels,dataset.data_infos[i]['location'])
        assert len(native)==len(rebuilt)==3 and all(torch.equal(a,b) for a,b in zip(native,rebuilt))
        checked.append(dict(scene_id=scene['scene_id'],frame_id=token,questions=labels,
                            exact_native_prompt_ids=[a.cpu().tolist() for a in native]))
    write(out/'native_input_validation.json',dict(status='PASS',checked=checked,
        timing='Preflight before all model requests, outside measured E2E. No model/LLM forward.'))
