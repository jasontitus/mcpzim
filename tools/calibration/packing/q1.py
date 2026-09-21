"""Prism62061f9 Q1_0 codec. Input scales are effective hard-candidate scales.

Never recompute scales from weights: GSQ learns them. GGML storage is FP16 d
followed by 16 little-bit-order sign bytes for each contiguous group of128.
"""
import numpy as np

GROUP = 128
BLOCK_BYTES = 18


def pack(sign_bits, scales, *, allow_scale_loss=False):
    bits = np.asarray(sign_bits)
    scales = np.asarray(scales, dtype=np.float32)
    if bits.dtype != np.uint8 or bits.ndim != 2 or bits.shape[1] % 16:
        raise ValueError('Expected uint8 signs [out,in/8], with in divisible by128')
    if scales.shape != (bits.shape[0], bits.shape[1] // 16) or not np.isfinite(scales).all():
        raise ValueError('Invalid group scale shape/value')
    with np.errstate(over='ignore', under='ignore'):
        half = scales.astype('<f2')
    if not np.isfinite(half).all():
        raise ValueError('Learned scale overflows FP16 storage')
    restored = half.astype(np.float32)
    if not allow_scale_loss and not np.array_equal(restored, scales):
        raise ValueError('Scale rounding/underflow would change hard candidate; explicit acceptance required')
    blocks = np.empty((*scales.shape, BLOCK_BYTES), dtype=np.uint8)
    blocks[..., :2] = half[..., None].view(np.uint8)
    blocks[..., 2:] = bits.reshape(*scales.shape, 16)
    delta = restored - scales
    return blocks.reshape(bits.shape[0], -1), {
        'max_scale_error': float(np.max(np.abs(delta), initial=0)),
        'sum_squared_weight_error': float(np.square(delta.astype(np.float64)).sum() * GROUP),
        'changed_groups': int(np.count_nonzero(delta)), 'bytes': int(blocks.nbytes)}


def unpack(packed):
    data = np.asarray(packed)
    if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] % BLOCK_BYTES:
        raise ValueError('Invalid Q1 block storage')
    blocks = data.reshape(data.shape[0], -1, BLOCK_BYTES)
    scales = np.ascontiguousarray(blocks[..., :2]).view('<f2').reshape(blocks.shape[:2]).astype(np.float32)
    if not np.isfinite(scales).all():
        raise ValueError('Nonfinite Q1 scale')
    signs = np.unpackbits(blocks[..., 2:], axis=-1, bitorder='little').astype(np.int8) * 2 - 1
    return (signs.astype(np.float32) * scales[..., None]).reshape(data.shape[0], -1)


def from_sign_logits(logits, effective_scales, **kwargs):
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] % GROUP or not np.isfinite(logits).all():
        raise ValueError('Invalid sign logits')
    # GSQ uses >0, unlike reference rounding's >=0. A zero logit is negative.
    return pack(np.packbits(logits > 0, axis=-1, bitorder='little'), effective_scales, **kwargs)


def v_head_permutation(k_heads, v_heads, head_dim):
    if k_heads <= 0 or v_heads % k_heads or head_dim <= 0:
        raise ValueError('Invalid recurrent head dimensions')
    return np.arange(v_heads * head_dim).reshape(k_heads, v_heads // k_heads, head_dim).transpose(1, 0, 2).reshape(-1)


def reorder_qwen_candidate(name, bits, scales, *, k_heads=16, v_heads=48, head_k_dim=128, head_v_dim=128):
    """Same grouped→tiled V-head layout as pinned converter, without dequantization."""
    bits, scales = np.asarray(bits), np.asarray(scales)
    if bits.ndim != 2 or scales.shape != (bits.shape[0], bits.shape[1] // 16):
        raise ValueError('Invalid candidate shapes')
    if '.linear_attn.' not in name or k_heads == v_heads:
        return bits, scales
    perm = v_head_permutation(k_heads, v_heads, head_v_dim)
    if name.endswith('.in_proj_qkv.weight'):
        qk = 2 * k_heads * head_k_dim
        rows = np.r_[np.arange(qk), perm + qk]
    elif name.endswith('.in_proj_z.weight'):
        rows = perm
    elif name.endswith(('.in_proj_a.weight', '.in_proj_b.weight')):
        rows = v_head_permutation(k_heads, v_heads, 1)
    elif name.endswith('.out_proj.weight'):
        if head_v_dim % GROUP:
            raise ValueError('V-head column permutation splits learned Q1 groups')
        groups = perm.reshape(-1, GROUP)
        if not np.array_equal(groups, groups[:, :1] + np.arange(GROUP)):
            raise ValueError('Noncontiguous learned scale groups')
        group_perm = groups[:, 0] // GROUP
        if bits.shape[1] != len(perm) // 8:
            raise ValueError('Output projection width mismatch')
        return bits.reshape(bits.shape[0], -1, 16)[:, group_perm].reshape(bits.shape), scales[:, group_perm]
    else:
        return bits, scales
    if len(rows) != bits.shape[0]:
        raise ValueError('Projection row shape mismatch')
    return bits[rows], scales[rows]
