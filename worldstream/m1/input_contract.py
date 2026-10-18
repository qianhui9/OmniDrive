"""Frame binding and GT isolation at the online API, without model imports."""
import hashlib
import numpy as np


def reject_oracle_metadata(value):
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).lower()
            if (name in {'gt', 'ground_truth', 'oracle', 'annotation', 'annotations',
                         'instance_token', 'instance_id', 'gt_mask', 'gt_boxes', 'gt_box'}
                    or name.startswith('gt_')):
                raise ValueError('Offline oracle information is forbidden in online metadata: ' + name)
            reject_oracle_metadata(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            reject_oracle_metadata(item)


def camera_layout(frame):
    return tuple((cid, c.coordinate_space, c.image.shape) for cid, c in frame.cameras.items())


def frame_digest(frame):
    """Bind commit to the exact observations/calibration verified at precheck."""
    reject_oracle_metadata(frame.metadata)
    digest = hashlib.sha256()
    def add(value):
        if value is None:
            digest.update(b'None;')
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, str(array.dtype))).encode())
            digest.update(array.tobytes())
    digest.update(repr((frame.scene_id, frame.frame_id, frame.timestamp, frame.execution_signature)).encode())
    add(frame.ego_to_world)
    for cid, camera in frame.cameras.items():
        image = np.asarray(camera.image)
        if image.dtype != np.uint8 or image.ndim not in (2, 3) or min(image.shape[:2]) < 1:
            raise ValueError('Expected nonempty uint8 camera image')
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError('Expected RGB camera image')
        digest.update(repr((cid, camera.camera_id, camera.timestamp, camera.coordinate_space,
                            camera.timing_reliable)).encode())
        for value in (image, camera.intrinsic, camera.world_to_camera, camera.raw_to_image,
                      camera.visibility_mask, camera.geometry_support_mask):
            add(value)
    return digest.hexdigest()
