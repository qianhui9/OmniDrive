"""One targeted v1 failure replay with the CURRENT v2 verifier; old data read-only."""
import argparse
from pathlib import Path
from tools.worldstream_m123.speculative_numeric_probe import probe


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/m3_world_guided_decode_run02'))
    root=parser.parse_args().root
    target=root/'numeric_replay';target.mkdir(exist_ok=False)
    (target/'native_profile').symlink_to('../../m3_world_guided_decode_run01/native_profile',target_is_directory=True)
    (target/'strict_gate_failure').symlink_to('../../m3_world_guided_decode_run01/strict_gate_failure',target_is_directory=True)
    probe(target,schedule=True)
