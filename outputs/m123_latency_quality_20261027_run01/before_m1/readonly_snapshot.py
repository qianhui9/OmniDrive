"""Owned immutable computation records, shared by M2 and cache publication.

Snapshot construction detaches mutable M1 containers once. Immutable leaves
(including compressed mask strings) are shared. deepcopy of a frozen subtree
is identity: neither M1 commit nor M2 consumers can mutate its containers.
Dense RGB correspondence arrays are never included in this interface.
"""
import numpy as np


def _readonly(*args, **kwargs):
    raise TypeError('read-only M1 computation snapshot')


class ReadOnlyDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __deepcopy__(self, memo):
        return self


class ReadOnlyList(list):
    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = reverse = sort = __iadd__ = __imul__ = _readonly

    def __deepcopy__(self, memo):
        return self


def freeze(value):
    if isinstance(value, (ReadOnlyDict, ReadOnlyList)):
        return value
    if isinstance(value, dict):
        return ReadOnlyDict((key, freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return ReadOnlyList(freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return value
    # Computation records must remain plain metadata, never mutable arrays or
    # arbitrary producer objects with hidden aliases.
    raise TypeError('Unsupported mutable M1 computation metadata: '+type(value).__name__)
