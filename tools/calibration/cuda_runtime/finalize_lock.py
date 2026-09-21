"""Remove CUDA packages already supplied by immutable base, validating versions."""
import re
from pathlib import Path


def main(root):
    protected = {}
    for line in (root / 'base-cuda-constraints.txt').read_text().splitlines():
        name, version = line.split('==')
        protected[name.lower()] = version
    text = (root / 'requirements.full.lock').read_text()
    blocks = re.split(r'(?=^[A-Za-z0-9][A-Za-z0-9_.-]*==)', text, flags=re.M)
    output = ['# CUDA/torch/triton provided by digest-pinned base; install with --no-deps.\n']
    for block in blocks:
        match = re.match(r'([A-Za-z0-9_.-]+)==([^\s\\]+)', block)
        if not match:
            continue
        name, version = match.groups()
        if name.lower() in protected:
            if protected[name.lower()] != version:
                raise ValueError(f'base CUDA version mismatch: {name}')
        elif name.lower().startswith(('cuda-', 'nvidia-')) or name.lower() in ('torch', 'triton'):
            raise ValueError(f'new unaccounted CUDA dependency: {name}')
        else:
            output.append(block)
    (root / 'requirements.lock').write_text(''.join(output))


if __name__ == '__main__':
    main(Path(__file__).parent)
