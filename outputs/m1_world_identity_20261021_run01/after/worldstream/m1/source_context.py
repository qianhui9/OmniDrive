"""Non-label scene/log/sample context supplied by the dataset or online owner.

A log may contain several scenes. Never synthesize log IDs from frame tokens.
This binds provenance only; it cannot qualify an identity producer.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType


@dataclass(frozen=True)
class LogSourceContext:
    scene_logs: object
    frame_scenes: object
    digest: str

    @classmethod
    def from_rows(cls, scenes, samples=()):
        scene_logs={};frame_scenes={}
        for row in scenes:
            scene,log=row['token'],row['log_token']
            if not isinstance(scene,str) or not scene or not isinstance(log,str) or not log:
                raise ValueError('Missing legitimate scene/log identifier')
            if scene in scene_logs:raise ValueError('Duplicate scene mapping')
            scene_logs[scene]=log
        for row in samples:
            token,scene=row['token'],row['scene_token']
            if token in frame_scenes or scene not in scene_logs:
                raise ValueError('Duplicate sample or unknown scene mapping')
            frame_scenes[token]=scene
        encoded=json.dumps(dict(scene_logs=scene_logs,frame_scenes=frame_scenes),sort_keys=True).encode()
        return cls(MappingProxyType(scene_logs),MappingProxyType(frame_scenes),hashlib.sha256(encoded).hexdigest())

    def log_for_scene(self, scene):
        if scene not in self.scene_logs:raise ValueError('Scene absent from trusted log context')
        return self.scene_logs[scene]

    def metadata_for(self, scene, frame, declared_log=None):
        log=self.log_for_scene(scene)
        if self.frame_scenes and self.frame_scenes.get(frame)!=scene:
            raise ValueError('Frame/scene mismatch in trusted source context')
        if declared_log is not None and declared_log!=log:
            raise ValueError('Dataset log does not match trusted source context')
        return dict(log_token=log,log_source=dict(scene_id=scene,log_token=log,
            context_sha256=self.digest,origin='trusted_dataset_scene_sample_tables'),log_provenance_status='bound')


def metadata_log_source(meta, scene, frame, context=None):
    if context is not None:return context.metadata_for(scene,frame,meta.get('log_token'))
    log=meta.get('log_token')
    if log is None:return dict(log_provenance_status='missing')
    if not isinstance(log,str) or not log.strip():raise ValueError('Invalid dataset log identifier')
    if meta.get('log_scene_token',scene)!=scene:raise ValueError('Dataset log/scene mismatch')
    return dict(log_token=log,log_source=dict(scene_id=scene,log_token=log,
        origin='dataset_metadata'),log_provenance_status='bound')


def log_binding_reason(metadata, scene):
    log=metadata.get('log_token');source=metadata.get('log_source')
    if log is not None and (not isinstance(log,str) or not log.strip()):return 'invalid_log_token'
    if source is not None and (not isinstance(source,dict) or source.get('scene_id')!=scene or source.get('log_token')!=log):
        return 'log_scene_binding_mismatch'
    return None
