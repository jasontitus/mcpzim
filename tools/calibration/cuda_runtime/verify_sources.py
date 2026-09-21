"""Fail closed if the staged upstream code differs from its source manifest."""
import hashlib
import json
from pathlib import Path

PINS = {
    'gsq': '03fc16484c369e3127225615d5e03e8d3a6043e3',
    'rco': '9a1e09c07d468109cbe60a1b87d5036034a79d10',
}


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if set(manifest) != set(PINS):
        raise ValueError('wrong source set')
    for name, revision in PINS.items():
        item = manifest[name]
        if item['revision'] != revision:
            raise ValueError('wrong source revision')
        actual = {str(path.relative_to(root / name)) for path in (root / name).rglob('*')
                  if path.is_file() and '__pycache__' not in path.parts}
        if actual != set(item['files']):
            raise ValueError('unexpected or missing source files')
        for relative, expected in item['files'].items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe manifest path')
            target = root / name / path
            if target.is_symlink() or not target.is_file() or any(
                    parent.is_symlink() for parent in target.parents if parent != root):
                raise ValueError(f'missing/nonregular source: {relative}')
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f'source checksum failed: {relative}')
    return {name: item['revision'] for name, item in manifest.items()}


if __name__ == '__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]), sort_keys=True))
