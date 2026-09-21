"""CPU-only, generation-pinned restoration of the reviewed f13 GSQ block2 source.

No model loading, CUDA calls, resource provisioning or identity migration. All
artifacts are published locally only after source hashes and identities verify.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from solver.gsq_migration import (canonical,sha,validate_source_receipt,validate_source_directory)


def prepare(source_receipt,expected_sha256,output,*,store_factory=None):
    source_receipt=Path(source_receipt);output=Path(output)
    if source_receipt.is_symlink() or source_receipt.stat().st_size>2*1024**2:
        raise ValueError('Unsafe source receipt')
    raw=source_receipt.read_bytes()
    if sha(raw)!=expected_sha256:raise ValueError('Source receipt differs from reviewed file hash')
    receipt=json.loads(raw);manifest=validate_source_receipt(receipt)
    if output.exists() or output.is_symlink():raise FileExistsError('Source destination must be new')
    if not output.parent.is_dir() or output.parent.is_symlink():raise ValueError('Existing ordinary parent required')
    required=sum(v['bytes'] for v in manifest['payloads'].values())+manifest['state_bundle']['bytes']+10*1024**3
    if shutil.disk_usage(output.parent).free<required:raise ValueError('Insufficient source restore disk headroom')
    if store_factory is None:
        from gcs_checkpoints import GCSCheckpointStore
        store_factory=GCSCheckpointStore
    pending=Path(tempfile.mkdtemp(prefix='.gsq24-source-',dir=output.parent))
    try:
        store=store_factory(manifest['bucket'],manifest['prefix'],project='tiltastech-zimfo',staging_dir=str(pending))
        restored=store.restore(receipt['snapshot'],manifest['identity'],pending/'payloads',
                              commit_generation=receipt['receipt']['commit']['generation'])
        if restored!=receipt['receipt']:raise ValueError('Remote source commit differs from reviewed receipt')
        validation=validate_source_directory(pending/'payloads',receipt)
        files={'source-receipt.json':raw,'source-manifest.json':canonical(manifest),
               'source-validation.json':canonical(validation)}
        for name,data in files.items():
            with (pending/name).open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
        fd=os.open(pending,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if output.exists() or output.is_symlink():raise FileExistsError('Source destination appeared during restore')
        os.rename(pending,output)
        fd=os.open(output.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if pending.exists():shutil.rmtree(pending)
    return {'status':'gsq24_source_prepared','source_commit':receipt['receipt']['commit'],
            'output':str(output),'source_receipt_sha256':sha(raw),'required_free_bytes':required,
            'source_validation_sha256':sha(canonical(validation)),'gpu_validation_passed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-receipt',required=True)
    parser.add_argument('--source-receipt-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source_receipt,args.source_receipt_sha256,args.output),indent=2))


if __name__=='__main__':main()
