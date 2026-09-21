import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upstream_check import validate_budget
from verify_sources import PINS, verify


@pytest.mark.parametrize('target', [float('nan'), float('inf'), 0.5, 17.0])
def test_reject_impossible_budget(target):
    with pytest.raises(RuntimeError):
        validate_budget(torch.tensor([1.125, 2.125, 16.]), torch.tensor([1.]), target)


def test_reject_bad_weights():
    with pytest.raises(RuntimeError):
        validate_budget(torch.tensor([1., 2.]), torch.tensor([-1.]), 1.5)


def source_fixture(root):
    manifest = {}
    for name, revision in PINS.items():
        (root / name).mkdir()
        (root / name / 'module.py').write_text('verified source')
        manifest[name] = {'revision': revision, 'files': {
            'module.py': hashlib.sha256(b'verified source').hexdigest()}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_corruption_rejected(tmp_path):
    source_fixture(tmp_path)
    assert verify(tmp_path) == PINS
    (tmp_path / 'gsq/module.py').write_text('modified source')
    with pytest.raises(ValueError, match='checksum'):
        verify(tmp_path)


def test_wrong_revision_rejected(tmp_path):
    manifest = source_fixture(tmp_path)
    manifest['rco']['revision'] = 'main'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='revision'):
        verify(tmp_path)


def test_path_escape_rejected(tmp_path):
    manifest = source_fixture(tmp_path)
    manifest['rco']['files']['../private'] = 'ignored'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_untracked_source_rejected(tmp_path):
    source_fixture(tmp_path)
    (tmp_path / 'rco/evil.py').write_text('unexpected module')
    with pytest.raises(ValueError, match='unexpected'):
        verify(tmp_path)
