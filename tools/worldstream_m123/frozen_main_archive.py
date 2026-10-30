"""Lossless chunking of NEW benchmark receipts; historical outputs are untouched."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .benchmark.common import sha256,write
from .frozen_main_benchmark import ROOT,read


def split_file(source,folder,paired=False):
    folder.mkdir(exist_ok=False)
    original_sha=sha256(source);archives=[];chunks={};total=0;index=[]
    # Paired records are ordered by pair then frame. Raw records are ordered by
    # pair/arm/frame. Each chunk remains small enough for ordinary Git storage.
    current=None;gz=None;fd=None;raw_hash=None;count=0;archive=None
    def finish():
        nonlocal gz,fd
        if gz is None:return
        gz.close();fd.close()
        archives.append(dict(archive=str(archive.relative_to(source.parent)),
            archive_sha256=sha256(archive),raw_sha256=raw_hash.hexdigest(),records=count))
    with source.open('rb') as handle:
        for line in handle:
            item=json.loads(line);key=item['pair_id']+('' if paired else '_'+item['condition'])
            if key!=current:
                finish();current=key;archive=folder/(key+'.jsonl.gz')
                fd=archive.open('xb');gz=gzip.GzipFile(filename='',mode='wb',fileobj=fd,mtime=0)
                raw_hash=hashlib.sha256();count=0
            gz.write(line);raw_hash.update(line)
            if paired:index.append(dict(pair_id=item['pair_id'],workload_id=item['workload_id'],
                frame_id=item['frame_id'],log_id=item['log_id'],receipt_archive=str(archive.relative_to(source.parent)),
                archive_row=count,record_sha256=hashlib.sha256(line).hexdigest(),format='lossless_paired_receipt_index_v1'))
            count+=1;total+=1
    finish()
    # Verify full concatenated original bytes before replacing any working copy.
    joined=hashlib.sha256();verified=0
    for entry in archives:
        h=hashlib.sha256()
        with gzip.open(source.parent/entry['archive'],'rb') as stream:
            for line in stream:joined.update(line);h.update(line);verified+=1
        assert h.hexdigest()==entry['raw_sha256']
    assert joined.hexdigest()==original_sha and total==verified
    if paired:
        temporary=source.with_suffix('.index.tmp')
        with temporary.open('w') as f:
            for item in index:f.write(json.dumps(item)+'\n')
        temporary.replace(source)
    else:source.unlink()
    return dict(original_path=str(source),original_sha256=original_sha,records=total,files=archives,
        retained_as='JSONL index plus exact-byte gzip chunks' if paired else 'exact-byte gzip chunks',
        roundtrip_verified=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    root=args.root;assert read(root/'main/run_status.json')['status']=='COMPLETED'
    assert read(root/'integrity_validation.json')['status']=='PASS'
    assert not (root/'receipt_archive.json').exists()
    primary=root/'paired_gpu_receipts.jsonl';duplicate=root/'main/paired_gpu_receipts.jsonl'
    assert sha256(primary)==sha256(duplicate)
    result=split_file(primary,root/'receipt_chunks',paired=True)
    write(root/'receipt_archive.json',dict(format='lossless_paired_receipt_index_v1',primary=result,
        duplicate_main_paired_sha256=sha256(duplicate),historical_outputs_touched=False))
    duplicate.unlink()
    raw=split_file(root/'main/raw_gpu_receipts.jsonl',root/'main/raw_receipt_chunks')
    manifest=read(root/'receipt_archive.json');manifest['raw_execution_order']=raw;write(root/'receipt_archive.json',manifest)
    print('LOSSLESS_RECEIPTS_CHUNKED',result['records'],raw['records'])


if __name__=='__main__':main()
