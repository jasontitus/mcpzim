"""Verify original BF16 shards against pinned public hashes and tensor headers.

Streams hashes from disk; never loads the model into GPU or host tensor memory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from encoding_inventory import REVISION


def verify(root, hashes, inventory, output):
    root, output = Path(root), Path(output)
    result = {"status": "validating", "model": hashes.get("model"), "revision": REVISION,
              "precision": "BF16", "verified_files": {}, "tensor_count": 0}

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output)

    save()
    try:
        if hashes["revision"] != REVISION or inventory["revision"] != REVISION:
            raise ValueError("Wrong original checkpoint revision")
        if hashes["model"] != "Qwen/Qwen3.8-27B" or inventory["model"] != hashes["model"]:
            raise ValueError("Wrong original model")
        config = json.loads((root / "config.json").read_text())
        if "quantization_config" in config or "quantization_config" in config.get("text_config", {}):
            raise ValueError("Quantized baseline configuration")
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        if set(index.values()) != set(hashes["files"]) or set(index) != set(inventory["tensors"]):
            raise ValueError("Index/expected tensor coverage mismatch")
        seen = set()
        for filename, expected in sorted(hashes["files"].items()):
            if Path(filename).name != filename:
                raise ValueError("Unsafe shard filename")
            path = root / filename
            if path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Wrong shard size: {filename}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != expected["sha256"]:
                    raise ValueError(f"Original hash mismatch: {filename}")
                stream.seek(0)
                header_length = struct.unpack("<Q", stream.read(8))[0]
                if not 2 <= header_length <= 2 * 1024**2:
                    raise ValueError("Unreasonable header size")
                header = json.loads(stream.read(header_length))
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                if name in seen or index.get(name) != filename or tensor != inventory["tensors"].get(name):
                    raise ValueError(f"Unexpected tensor: {name}")
                if tensor["dtype"] != "BF16":
                    raise ValueError(f"Not BF16: {name}")
                begin, end = tensor["data_offsets"]
                if end - begin != math.prod(tensor["shape"]) * 2 or not 0 <= begin < end <= expected["bytes"] - 8 - header_length:
                    raise ValueError(f"Invalid tensor storage: {name}")
                seen.add(name)
            result["verified_files"][filename] = expected
            result["tensor_count"] = len(seen)
            save()
            print(f"Verified {len(result['verified_files'])}/{len(hashes['files'])} original shards", flush=True)
        if seen != set(index):
            raise ValueError("Missing tensors")
        result["status"] = "validated"
        result["weight_file_bytes"] = sum(f["bytes"] for f in hashes["files"].values())
        save()
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        save()
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hashes", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.model_dir, json.loads(args.hashes.read_text()),
           json.loads(args.inventory.read_text()), args.output)
