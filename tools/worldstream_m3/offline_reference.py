"""Read unchanged native evaluation annotations; never imported by online M3."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=1)
def description_questions():
    tree=ast.parse(Path('projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoadAnnoatationVQATest')
    for node in ast.walk(cls):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='template' for t in node.targets):
            return frozenset(ast.literal_eval(node.value))
    return frozenset()


@lru_cache(maxsize=256)
def description_annotation(token):
    path=Path('data/nuscenes/desc/val')/(token+'.json')
    if not path.exists():return None
    raw=path.read_bytes();value=json.loads(raw)
    text=value.get('description')
    return dict(text=text,source=str(path),sha256=hashlib.sha256(raw).hexdigest(),kind='original_native_description_annotation') if isinstance(text,str) else None


def resolve_reference(token,question,raw_targets):
    if question in raw_targets:
        return raw_targets[question],dict(kind='original_conv_or_vqa_annotation',sample_token=token)
    if question in description_questions():
        annotation=description_annotation(token)
        if annotation:return annotation['text'],{k:v for k,v in annotation.items() if k!='text'}
    # desc/action is a longer rationale, not the missing validation keywords
    # target for the short-action instruction. Do not silently substitute it.
    return None,None
