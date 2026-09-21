"""Bounded Q1 candidate I/O. Bits encode positive=1, column zero in low bit."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def pack_signs(signs):
    if signs.ndim != 2 or signs.shape[1] % 128:
        raise ValueError('Q1 requires two dimensions and 128-column groups')
    if not torch.isfinite(signs).all():
        raise ValueError('Nonfinite candidate logits/signs')
    bits = (signs > 0).to(torch.uint8).reshape(signs.shape[0], -1, 8)
    shifts = torch.arange(8, device=bits.device, dtype=torch.uint8)
    return (bits << shifts).sum(-1).to(torch.uint8)


def save_candidate(path, signs, scales):
    if Path(path).exists():
        raise FileExistsError('Candidate files are immutable')
    codes = pack_signs(signs).cpu().contiguous()
    effective = scales.to(torch.bfloat16).cpu().contiguous()
    if effective.shape != (signs.shape[0], signs.shape[1] // 128):
        raise ValueError('Wrong group scale shape')
    if not torch.isfinite(effective.float()).all():
        raise ValueError('Nonfinite candidate scales')
    save_file({'codes': codes, 'scales': effective}, str(path),
              metadata={'format': 'zimfo-q1-v1', 'group_size': '128', 'bit_order': 'little',
                        'effective_scale_dtype': 'BF16', 'zero_logit_sign': 'negative'})


class Q1Candidate:
    def __init__(self, path):
        self.path = Path(path)
        with safe_open(str(path), framework='pt') as f:
            metadata = f.metadata()
            if metadata.get('format') != 'zimfo-q1-v1' or metadata.get('group_size') != '128' or metadata.get('bit_order') != 'little':
                raise ValueError('Unsupported candidate representation')
            codes = f.get_slice('codes')
            if len(codes.get_shape()) != 2 or codes.get_dtype() != 'U8':
                raise ValueError('Q1 codes require a uint8 matrix')
            self.shape = (codes.get_shape()[0], codes.get_shape()[1] * 8)
            if not all(self.shape) or self.shape[1] % 128 or f.get_slice('scales').get_dtype() != 'BF16' or f.get_slice('scales').get_shape() != [self.shape[0], self.shape[1] // 128]:
                raise ValueError('Candidate shapes disagree')
        self.packed_weight_bytes = self.shape[0] * self.shape[1] // 128 * 18
        self.identity = self._stat()
        digest = hashlib.sha256()
        with self.path.open('rb') as f:
            for block in iter(lambda: f.read(8*1024**2), b''):
                digest.update(block)
        self.sha256 = digest.hexdigest()

    def _stat(self):
        s = self.path.stat()
        return (s.st_ino, s.st_size, s.st_mtime_ns)

    def check_immutable(self):
        if self._stat() != self.identity:
            raise ValueError('Candidate changed after content verification')

    def rows(self, start, end, device, dtype):
        self.check_immutable()
        if not 0 <= start < end <= self.shape[0]:
            raise ValueError('Invalid candidate row slice')
        with safe_open(str(self.path), framework='pt') as f:
            codes = f.get_slice('codes')[start:end].to(device)
            scales = f.get_slice('scales')[start:end].to(device=device, dtype=dtype)
        shifts = torch.arange(8, device=device, dtype=torch.uint8)
        bits = ((codes.unsqueeze(-1) >> shifts) & 1).reshape(end-start, self.shape[1])
        return (bits.to(dtype) * 2 - 1) * scales.repeat_interleave(128, dim=1)


class StreamedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, reference, probability, candidate, rows):
        if probability.ndim != 0 or rows <= 0 or reference.device != x.device or reference.dtype != x.dtype or probability.device != x.device:
            raise ValueError('Scalar probability, positive row chunk and matching device/dtype required')
        if reference.requires_grad or tuple(reference.shape) != candidate.shape:
            raise ValueError('Frozen reference shape must match candidate')
        result = x.new_empty((*x.shape[:-1], reference.shape[0]))
        for start in range(0, reference.shape[0], rows):
            end = min(start + rows, reference.shape[0])
            ref = reference[start:end]
            delta = candidate.rows(start, end, x.device, torch.float32) - ref.float()
            mixed = (ref.float() + probability.float() * delta).to(x.dtype)
            result[..., start:end] = torch.nn.functional.linear(x, mixed)
        ctx.save_for_backward(x, reference, probability)
        ctx.candidate, ctx.rows = candidate, rows
        return result

    @staticmethod
    def backward(ctx, grad):
        x, reference, probability = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1]).float()
        grad_flat = grad.reshape(-1, grad.shape[-1]).float()
        gx = torch.zeros_like(flat)
        gp = torch.zeros_like(probability, dtype=torch.float32)
        for start in range(0, reference.shape[0], ctx.rows):
            end = min(start + ctx.rows, reference.shape[0])
            ref = reference[start:end]
            delta = ctx.candidate.rows(start, end, x.device, torch.float32) - ref.float()
            mixed = (ref.float() + probability.float() * delta).to(x.dtype).float()
            g = grad_flat[:, start:end]
            gx.add_(g @ mixed)
            gp.add_(((g.T @ flat).to(x.dtype).float() * delta).sum())
        return gx.reshape_as(x).to(x.dtype), None, gp.to(probability.dtype), None, None
