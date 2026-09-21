"""Small streaming GGUF v3 serializer with exact on-disk byte accounting.

Accepts metadata copied from a trusted pinned-converter GGUF template. Tensor
payloads must already have runtime names/layout; q1.py handles sign/scale blocks.
No hidden rounding, quantization, or architecture inference occurs here.
"""
import hashlib
import os
from pathlib import Path
import struct

ALIGNMENT = 32
TYPES = {0: (1, 4), 1: (1, 2), 30: (1, 2), 41: (128, 18)}  # F32,F16,PrismQ1_0


def string(text):
    data = text.encode('utf-8')
    return struct.pack('<Q', len(data)) + data


def align(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def tensor_info(name, shape, kind, offset):
    if kind not in TYPES or not shape or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('Unsupported tensor type/shape')
    group, size = TYPES[kind]
    if shape[-1] % group:
        raise ValueError('Tensor row does not align to quantization group')
    return string(name) + struct.pack('<I', len(shape)) + struct.pack('<'+'Q'*len(shape), *reversed(shape)) + struct.pack('<IQ', kind, offset)


def expected_nbytes(shape, kind):
    tensor_info('check', shape, kind, 0)
    import math
    group, size = TYPES[kind]
    return math.prod(shape) // group * size


def validate_metadata(data, count):
    """Parse exactly the declared KV section; enforce this writer's alignment."""
    if type(count) is not int or not 0 <= count <= 1000000:
        raise ValueError('Invalid metadata count')
    offset=0; values={}
    sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def take(n):
        nonlocal offset
        if n < 0 or offset+n > len(data): raise ValueError('Truncated metadata')
        result=data[offset:offset+n];offset+=n;return result
    def uint(fmt): return struct.unpack(fmt,take(struct.calcsize(fmt)))[0]
    def read_string(): return take(uint('<Q')).decode('utf-8')
    def value(kind, depth=0):
        if depth>2: raise ValueError('Nested metadata arrays unsupported')
        if kind==8: return read_string()
        if kind==9:
            subtype=uint('<I');n=uint('<Q')
            if subtype in sizes: take(n*sizes[subtype]);return None
            if n>len(data): raise ValueError('Invalid metadata array length')
            for _ in range(n): value(subtype,depth+1)
            return None
        if kind not in sizes: raise ValueError('Unsupported metadata type')
        raw=take(sizes[kind])
        return int.from_bytes(raw,'little') if kind in (0,2,4,10) else None
    for _ in range(count):
        key=read_string()
        if key in values: raise ValueError('Duplicate metadata key')
        values[key]=value(uint('<I'))
    if offset!=len(data): raise ValueError('Metadata count/trailing bytes disagree')
    if values.get('general.alignment',32)!=ALIGNMENT: raise ValueError('Unsupported GGUF alignment')
    return values


def serialized_size(metadata_bytes, metadata_count, tensors):
    validate_metadata(metadata_bytes,metadata_count)
    offset=0; length=24+len(metadata_bytes)
    for t in tensors:
        length+=len(tensor_info(t['name'],t['shape'],t['type'],offset))
        offset=align(offset+expected_nbytes(t['shape'],t['type']))
    return align(length)+offset


def write(path, metadata_bytes, metadata_count, tensors):
    """tensors entries: name, shape in numpy order, type, payload path, sha256.

    Metadata bytes exclude the GGUF header. Caller must use alignment32 metadata.
    Atomic publish only after checking every declared payload size and digest.
    """
    validate_metadata(metadata_bytes, metadata_count)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    names = [t['name'] for t in tensors]
    if not tensors or len(set(names)) != len(names):
        raise ValueError('Empty or duplicate tensor list')
    offset, infos = 0, []
    for t in tensors:
        size = expected_nbytes(t['shape'], t['type'])
        if Path(t['payload']).stat().st_size != size:
            raise ValueError('Payload bytes do not match shape/type')
        infos.append(tensor_info(t['name'], t['shape'], t['type'], offset))
        offset = align(offset + size)
    header = struct.pack('<4sIQQ', b'GGUF', 3, len(tensors), metadata_count)
    prefix = header + metadata_bytes + b''.join(infos)
    expected_bytes = align(len(prefix)) + offset
    temporary = path.with_name(path.name + '.partial')
    owned = False
    try:
        with temporary.open('xb') as stream:
            owned = True
            stream.write(prefix)
            stream.write(bytes(align(stream.tell()) - stream.tell()))
            for tensor in tensors:
                digest = hashlib.sha256()
                with Path(tensor['payload']).open('rb') as payload:
                    while chunk := payload.read(1024**2):
                        digest.update(chunk)
                        stream.write(chunk)
                if digest.hexdigest() != tensor['sha256']:
                    raise ValueError('Payload hash mismatch')
                stream.write(bytes(align(stream.tell()) - stream.tell()))
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != expected_bytes:
            raise ValueError('Serialized size accounting mismatch')
        os.link(temporary, path)  # exclusive publish, cannot replace another writer
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except BaseException:
        if owned: temporary.unlink(missing_ok=True)
        raise
    return {'bytes': expected_bytes, 'tensor_payload_bytes': sum(expected_nbytes(t['shape'], t['type']) for t in tensors),
            'tensor_count': len(tensors), 'alignment': ALIGNMENT}
