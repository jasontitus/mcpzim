"""Smoke bench — run a single GGUF end-to-end via llama-cpp-python.

Reports prefill tok/s, decode tok/s, peak RSS. Matches the output format
of ../llm-smoke/bench.py (MLX) so side-by-side comparison in a spreadsheet
is trivial.

Usage:
  .venv/bin/python bench.py --repo mlx-community/gemma-3-4b-it-gguf \\
    --file gemma-3-4b-it-Q4_K_M.gguf \\
    --prompt-tokens 5000 --max-new-tokens 128 \\
    --n-gpu-layers -1 --cache-type-k q8_0 --cache-type-v q8_0

--n-gpu-layers -1 offloads every layer to Metal. Leave --cache-type-*
at their default f16 for the first-pass memory measurement, then rerun
with q8_0/q4_0 to see the KV-quant savings.
"""

import argparse
import gc
import os
import resource
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import psutil
from huggingface_hub import hf_hub_download


def rss_mb() -> float:
    """Resident set size in MB, platform-safe."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes. llama.cpp memory
    # lives in Metal's unified memory pool which shows up in RSS on
    # macOS, so this is a reasonable peak indicator.
    if sys.platform == "darwin":
        return r / (1024 * 1024)
    return r / 1024


@dataclass
class MemoryProbe:
    samples: list[float]
    stop: threading.Event

    def start(self, interval_s: float = 0.1):
        def loop():
            proc = psutil.Process()
            while not self.stop.is_set():
                # physical_footprint is macOS-specific via Apple's memorystatus;
                # psutil's rss is a decent stand-in that tracks GPU unified
                # memory too. ru_maxrss captures the peak.
                self.samples.append(proc.memory_info().rss / (1024 * 1024))
                time.sleep(interval_s)
        threading.Thread(target=loop, daemon=True).start()

    @classmethod
    def create(cls) -> "MemoryProbe":
        return cls(samples=[], stop=threading.Event())

    def peak(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def count_ge(self, mb: float) -> int:
        return sum(1 for s in self.samples if s >= mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="HuggingFace repo id holding the GGUF")
    ap.add_argument("--file", required=True,
                    help="GGUF filename inside the repo")
    ap.add_argument("--prompt-tokens", type=int, default=5000,
                    help="Rough target prompt length in tokens")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--n-ctx", type=int, default=8192,
                    help="llama.cpp context window")
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="Number of layers to offload to Metal; -1 = all")
    ap.add_argument("--cache-type-k", default="f16",
                    choices=["f16", "f32", "q8_0", "q5_1", "q5_0",
                             "q4_1", "q4_0", "iq4_nl"])
    ap.add_argument("--cache-type-v", default="f16",
                    choices=["f16", "f32", "q8_0", "q5_1", "q5_0",
                             "q4_1", "q4_0", "iq4_nl"])
    ap.add_argument("--flash-attn", action="store_true",
                    help="Enable flash attention (required for quantized KV)")
    ap.add_argument("--swa-full", choices=["true", "false", "default"],
                    default="default",
                    help="iSWA cache mode. 'false' enables rotation-based "
                         "pruning (PR #13194/#21513) — prior SWA tokens get "
                         "rotated out of the sliding window cache so cache "
                         "size stays bounded. 'true' keeps the full SWA "
                         "cache and disables pruning. 'default' uses the "
                         "model's own setting (usually full).")
    ap.add_argument("--prompt", default="",
                    help="Override prompt text. Empty = filler text sized "
                         "to --prompt-tokens.")
    args = ap.parse_args()

    # Import here so --help is snappy when the wheel's slow to import.
    from llama_cpp import Llama

    print(f"bench: repo={args.repo} file={args.file}")
    print(f"        n_ctx={args.n_ctx} n_gpu_layers={args.n_gpu_layers} "
          f"cache_k={args.cache_type_k} cache_v={args.cache_type_v} "
          f"flash_attn={args.flash_attn}")

    t0 = time.perf_counter()
    gguf_path = hf_hub_download(repo_id=args.repo, filename=args.file)
    print(f"        gguf path: {gguf_path} ({os.path.getsize(gguf_path)/1e9:.2f} GB)")
    print(f"        download/cache lookup: {time.perf_counter()-t0:.2f}s")
    print(f"        rss before load: {rss_mb():.0f} MB")

    probe = MemoryProbe.create()
    probe.start()

    t_load = time.perf_counter()
    swa_full_arg: Optional[bool] = None
    if args.swa_full == "true":
        swa_full_arg = True
    elif args.swa_full == "false":
        swa_full_arg = False
    llm = Llama(
        model_path=gguf_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        type_k=_kv_type(args.cache_type_k),
        type_v=_kv_type(args.cache_type_v),
        flash_attn=args.flash_attn,
        swa_full=swa_full_arg,
        verbose=False,
    )
    load_s = time.perf_counter() - t_load
    print(f"        load: {load_s:.2f}s · rss now: {rss_mb():.0f} MB")

    prompt = args.prompt or _filler_prompt(args.prompt_tokens)
    # Tokenize once so we can report the true prompt length.
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    print(f"        prompt: {len(tokens)} tokens")

    # Prefill + decode via the raw token API so we can time prefill vs decode
    # independently — the high-level create_completion conflates them.
    t_prefill = time.perf_counter()
    llm.reset()
    llm.eval(tokens)
    prefill_s = time.perf_counter() - t_prefill

    t_decode = time.perf_counter()
    decoded_n = 0
    decoded_tokens: list[int] = []
    for _ in range(args.max_new_tokens):
        tok = llm.sample()
        if tok == llm.token_eos():
            break
        decoded_tokens.append(tok)
        llm.eval([tok])
        decoded_n += 1
    decode_s = time.perf_counter() - t_decode

    probe.stop.set()
    gc.collect()

    prefill_tps = len(tokens) / prefill_s if prefill_s > 0 else 0
    decode_tps = decoded_n / decode_s if decode_s > 0 else 0
    peak_mb = max(probe.peak(), rss_mb())

    print()
    print(f"RESULT prompt_tokens={len(tokens)} new_tokens={decoded_n}")
    print(f"RESULT prefill_s={prefill_s:.2f} decode_s={decode_s:.2f}")
    print(f"RESULT prefill_tps={prefill_tps:.0f} decode_tps={decode_tps:.1f}")
    print(f"RESULT peak_mb={peak_mb:.0f} "
          f"ge5gb={probe.count_ge(5000)} ge6gb={probe.count_ge(6000)} "
          f"ge7gb={probe.count_ge(7000)} samples={len(probe.samples)}")
    sample = llm.detokenize(decoded_tokens).decode("utf-8", errors="replace")
    print(f"RESULT sample_decode={sample[:200]!r}")


def _kv_type(name: str) -> int:
    """Map llama.cpp ggml type name → numeric enum.

    These IDs come from `enum ggml_type` in ggml.h; llama-cpp-python's
    Python bindings don't re-export them so we hard-code the stable
    subset we care about for the KV cache.
    """
    mapping = {
        "f32":    0,
        "f16":    1,
        "q4_0":   2,
        "q4_1":   3,
        "q5_0":   6,
        "q5_1":   7,
        "q8_0":   8,
        "iq4_nl": 20,
    }
    return mapping[name]


def _filler_prompt(target_tokens: int) -> str:
    """Filler prompt sized to ~target_tokens. Same approach as the
    MLX bench_memory.py — paragraphs of English prose tokenize at
    ~4 chars/token so 4x target gets us in the ballpark. The prompt
    content doesn't matter for memory measurement; we just need to
    fill the KV cache."""
    para = (
        "The rain in Spain falls mainly on the plain. "
        "In fact, the fire in the heart of the evening sun is blinding. "
        "Tokens are the fundamental unit of language modeling. "
        "The quick brown fox jumps over the lazy dog in a hundred ways. "
    )
    # ~60 chars/para → ~15 tokens. Repeat until we have enough.
    copies = max(1, (target_tokens * 5) // len(para))
    return (para * copies).strip()


if __name__ == "__main__":
    main()
