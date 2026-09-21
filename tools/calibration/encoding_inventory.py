"""Read only public safetensors headers; compute illustrative Q1 storage costs.

No full weight download, quantization, runtime compatibility or quality claim.
The original checkpoint includes text, vision and MTP tensors; report separately.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import urllib.request

REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ROOT = f"https://huggingface.co/Qwen/Qwen3.8-27B/resolve/{REVISION}/"
BONSAI_BYTES = 3803452480


def read_range(url, start, length):
    request = urllib.request.Request(url + f"?header_offset={start}&header_length={length}",
                                     headers={"Range": f"bytes={start}-{start+length-1}"})
    with urllib.request.urlopen(request, timeout=40) as response:
        if response.status != 206:
            raise RuntimeError("Server ignored bounded range request; refusing full download")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{start+length-1}/"):
            raise RuntimeError(f"Unexpected content range: {content_range}")
        data = response.read(length + 1)
    if len(data) != length: raise RuntimeError("Wrong range length")
    return data


def header(filename):
    if not re.fullmatch(r"model-\d+-of-\d+\.safetensors", filename):
        raise ValueError("Unexpected shard filename")
    url = ROOT + filename
    length = struct.unpack("<Q", read_range(url, 0, 8))[0]
    if not 2 <= length <= 2 * 1024**2:
        raise ValueError("Unreasonable safetensors header size")
    raw = read_range(url, 8, length)
    return filename, json.loads(raw), hashlib.sha256(raw).hexdigest(), length + 8


def estimate(tensors, protected_controls):
    total, quantized, retained = 0, 0, 0
    for name, tensor in tensors.items():
        shape = tensor["shape"]
        n = math.prod(shape)
        protected = protected_controls and any(part in name for part in ("in_proj_a.", "in_proj_b."))
        if len(shape) == 2 and shape[1] % 128 == 0 and not protected:
            size = n // 128 * 18
            quantized += n
        else:
            size = n * 4  # keep small / unsupported tensors at F32 in this scenario
            retained += n
        total += ((size + 31) // 32) * 32
    return {"tensor_data_bytes_with_32byte_alignment": total,
            "q1_parameters": quantized, "f32_parameters": retained,
            "remaining_bytes_before_gguf_metadata": BONSAI_BYTES - total}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(ROOT + "model.safetensors.index.json", timeout=30) as response:
        index = json.load(response)
    shards = sorted(set(index["weight_map"].values()))
    tensors, hashes, downloaded = {}, {}, 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for filename, data, digest, byte_count in pool.map(header, shards):
            tensors.update({k: v for k, v in data.items() if k != "__metadata__"})
            hashes[filename] = digest
            downloaded += byte_count
    if set(tensors) != set(index["weight_map"]):
        raise ValueError("Header/index tensor coverage mismatch")
    if any(t["dtype"] != "BF16" for t in tensors.values()):
        raise ValueError("Unexpected baseline dtype")
    subsets = {"text": {}, "vision": {}, "mtp": {}, "other": {}}
    for name, tensor in tensors.items():
        category = ("text" if name.startswith("model.language_model.") or name == "lm_head.weight"
                    else "vision" if name.startswith("model.visual.")
                    else "mtp" if name.startswith("mtp.") else "other")
        subsets[category][name] = tensor
    if subsets["other"]: raise ValueError("Unclassified source tensors")
    result = {"model": "Qwen/Qwen3.8-27B", "revision": REVISION,
              "bonsai_comparison_bytes": BONSAI_BYTES,
              "header_bytes_downloaded": downloaded, "shard_header_sha256": hashes,
              "parameters_by_component": {k: sum(math.prod(t["shape"]) for t in v.values()) for k,v in subsets.items()},
              "text_q1_optimistic": estimate(subsets["text"], False),
              "text_q1_protected_recurrent_controls": estimate(subsets["text"], True),
              "all_checkpoint_q1_optimistic": estimate(tensors, False),
              "limitations": ["Q1 scenario includes embeddings and output head at one bit",
                              "Text subset excludes optional vision/MTP, not a different teacher",
                              "Excludes GGUF metadata/tokenizer and any required extra runtime assets",
                              "No assertion all these tensors can be optimized/loaded at Q1",
                              "No quality conclusion; protected controls may need higher precision"],
              "tensors": tensors}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("tensors", "shard_header_sha256")}, indent=2))
