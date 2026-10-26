"""Offline before-code profiling from a frozen source snapshot, never serving."""
import importlib.util
from pathlib import Path
import runpy
import sys
import worldstream


def main():
    if len(sys.argv)<3 or sys.argv[1]!='--snapshot-root':
        raise SystemExit('frozen_worker --snapshot-root DIR <worker arguments>')
    root=Path(sys.argv[2]);sys.argv=sys.argv[:1]+sys.argv[3:]
    for component in ('m1','m2','m3'):
        name='worldstream.'+component;directory=root/('before_'+component)
        if any(k==name or k.startswith(name+'.') for k in sys.modules):
            raise ValueError('Frozen runtime must load in a fresh process')
        spec=importlib.util.spec_from_file_location(name,directory/'__init__.py',submodule_search_locations=[str(directory)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        setattr(worldstream,component,module);spec.loader.exec_module(module)
    runpy.run_module('tools.worldstream_m123.benchmark.worker',run_name='__main__')


if __name__=='__main__':main()
