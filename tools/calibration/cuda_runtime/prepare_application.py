"""Allowlisted public/application code context; no credentials, weights or captures."""
import hashlib
import json
from pathlib import Path
import shutil
from prepare_sources import stage


def main():
    here=Path(__file__).parent
    app=here/'.context/application'
    if app.exists():shutil.rmtree(app)
    app.mkdir(parents=True)
    source=here.parent
    paths=[source/name for name in ['checkpoints.py','gcs_checkpoints.py','restore_inputs.py','validate_capture.py','verify_baseline.py','encoding_inventory.py','prepare_gsq24_source.py']]
    for folder in ['solver','packing']:
        paths.extend(p for p in (source/folder).rglob('*.py') if '__pycache__' not in p.parts)
    paths.append(source/'packing/evidence/qwen-cost-manifest.json')
    paths.extend(source/name for name in ('tests/test_gcs_checkpoints.py', 'tests/test_restore_inputs.py'))
    manifest={}
    for path in paths:
        relative=path.relative_to(source);target=app/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        manifest[str(relative)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    (app/'runtime-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    prism=here/'.context/prism'
    if not prism.exists():
        pin='62061f91088281e65071cc38c5f69ee95c39f14e'
        raw=stage(Path('/tmp/zimfo-prism-packing'),prism,pin)
        receipt={'revision':pin,'files':{name:{'sha256':value,'bytes':(prism/name).stat().st_size}
                                       for name,value in raw['files'].items()}}
        (prism/'zimfo-prism-source.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
