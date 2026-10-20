"""Choose a fresh result namespace without editing measured source code."""
import argparse
import importlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',required=True,choices=['profile','attribution','service','micro','prefix','confirmation','finalize'])
    p.add_argument('--output',required=True)
    args=p.parse_args()
    from . import profile
    profile.ROOT=Path(args.output)
    module=profile if args.phase=='profile' else importlib.import_module(__package__+'.'+args.phase)
    module.main()


if __name__=='__main__':main()
