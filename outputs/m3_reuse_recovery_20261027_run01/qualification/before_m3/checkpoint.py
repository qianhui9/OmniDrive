"""Attach the identity of the checkpoint just loaded by the standard launcher."""
import hashlib


def attach_checkpoint_identity(model, path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024**2),b''):digest.update(chunk)
    identity=dict(getattr(model,'worldstream_audit_identity',{}))
    identity.update(checkpoint=str(path),checkpoint_sha256=digest.hexdigest())
    model.worldstream_audit_identity=identity
    return identity
