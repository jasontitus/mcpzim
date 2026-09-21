"""Stage only immutable public source revisions; never copy a working tree."""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

PINS = {
    'gsq': '03fc16484c369e3127225615d5e03e8d3a6043e3',
    'rco': '9a1e09c07d468109cbe60a1b87d5036034a79d10',
}


def stage(source, destination, revision):
    resolved = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', revision + '^{commit}'], text=True
    ).strip()
    if resolved != revision:
        raise ValueError('source revision mismatch')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    raw = subprocess.check_output(['git', '-C', str(source), 'archive', revision])
    destination.mkdir(parents=True)
    records = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for entry in archive:
            path = Path(entry.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe archive path')
            if entry.isdir():
                continue
            if not entry.isfile():
                raise ValueError('only regular source files allowed')
            payload = archive.extractfile(entry).read()
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            records[entry.name] = hashlib.sha256(payload).hexdigest()
    if 'LICENSE' not in records:
        raise ValueError('upstream attribution missing')
    return {'revision': revision, 'files': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gsq', type=Path, required=True)
    parser.add_argument('--rco', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / '.context/sources')
    args = parser.parse_args()
    manifest = {name: stage(getattr(args, name), args.output / name, revision)
                for name, revision in PINS.items()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
