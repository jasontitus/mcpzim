"""LoRA fine-tune + fuse on CUDA (PEFT/transformers equivalent of mlx_lm lora + fuse).

Drop-in for the LoRA stage of finetune.sh on Linux/CUDA. Reads the same
mlx-lm-style data dir (train.jsonl + valid.jsonl with `messages` field),
applies LoRA to the last `--num-layers` transformer blocks, trains for
`--iters` steps, and saves merged FP16 weights to `--fused-path`.

Output layout matches what finetune.sh's downstream steps expect:
  $ADAPTERS_DIR/adapter_model.safetensors  (PEFT name; harmless that it
                                             differs from mlx-lm's
                                             `adapters.safetensors`)
  $ADAPTERS_DIR/adapter_config.json
  $FUSED_DIR/                               (full HF checkpoint, ready
                                             for convert_hf_to_gguf.py)

Usage:
  python finetune_cuda.py \
      --model google/gemma-3-4b-it \
      --data-dir out/data \
      --adapter-path out/adapters \
      --fused-path out/fused-hf \
      --iters 500 --num-layers 16 --batch-size 4 \
      --learning-rate 1e-5 --max-seq-length 2048 --val-batches 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def _is_multimodal(model_name: str) -> bool:
    """True if the HF config has a `text_config` sub-config (composite
    image-text-to-text model). Triggers the language_model-extraction path."""
    cfg = AutoConfig.from_pretrained(model_name)
    return getattr(cfg, "text_config", None) is not None


def load_base(model_name: str, dtype=torch.bfloat16):
    """Returns (causal_lm_module, full_model_or_None, tokenizer).

    For multimodal bases (e.g. gemma-3-4b-it), `causal_lm_module` is
    `full.language_model` — we LoRA only that subtree. `full` is kept so
    we can save the full multimodal checkpoint after merging if we want;
    for our pipeline we only need the text core (convert_hf_to_gguf.py
    converts `gemma3_text`).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map=auto/cuda has been unreliable here: Gemma 3 1B silently
    # stayed on CPU even with explicit `device_map={"": "cuda"}`. Skip
    # device_map entirely; load on CPU first, .to("cuda") afterward —
    # the move is fast (1B–4B fits in seconds) and unambiguous.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _is_multimodal(model_name):
        print(f"  multimodal base: loading via AutoModelForImageTextToText "
              f"and extracting .language_model")
        full = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device)
        return full.language_model, full, tokenizer

    full = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    return full, None, tokenizer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_row(tokenizer, row: dict, max_seq_length: int) -> dict:
    """Apply chat template + tokenize. Truncate to max_seq_length."""
    messages = row["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        return_tensors=None,
    )
    enc["labels"] = list(enc["input_ids"])
    return enc


def collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(b["labels"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def find_target_modules(model) -> list[str]:
    """Pick the standard set: attention + MLP linears. Works for Gemma 3,
    Qwen 2.5/3.x, Llama family. Returns names that exist on this model."""
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
    found = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            found.add(leaf)
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model name or local path (the FP base, e.g. "
                         "google/gemma-3-4b-it)")
    ap.add_argument("--data-dir", required=True,
                    help="dir containing train.jsonl + valid.jsonl")
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--fused-path", required=True)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--num-layers", type=int, default=16,
                    help="# of LAST transformer blocks to apply LoRA to "
                         "(matches mlx_lm.lora's --num-layers semantics)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--val-batches", type=int, default=5)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-accum", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_rows = load_jsonl(data_dir / "train.jsonl")
    valid_rows = load_jsonl(data_dir / "valid.jsonl")
    print(f"loaded train={len(train_rows)} valid={len(valid_rows)}")

    print(f"loading tokenizer + model: {args.model}")
    model, _full, tokenizer = load_base(args.model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    n_layers = model.config.num_hidden_layers
    layers_to_transform = list(range(max(0, n_layers - args.num_layers),
                                     n_layers))
    print(f"applying LoRA to last {args.num_layers}/{n_layers} blocks: "
          f"{layers_to_transform}")

    targets = find_target_modules(model)
    print(f"target_modules: {targets}")

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        layers_to_transform=layers_to_transform,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable/1e6:.2f}M / {total/1e6:.0f}M "
          f"({100*trainable/total:.3f}%)")

    print("tokenizing")
    train_ds = Dataset.from_list(train_rows).map(
        lambda r: render_row(tokenizer, r, args.max_seq_length),
        remove_columns=["messages"],
        desc="train",
    )
    valid_ds = Dataset.from_list(valid_rows).map(
        lambda r: render_row(tokenizer, r, args.max_seq_length),
        remove_columns=["messages"],
        desc="valid",
    )

    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device

    def make_loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate(b, pad_id),
            num_workers=0,
            drop_last=shuffle,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    valid_loader = make_loader(valid_ds, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    warmup = max(1, int(args.warmup_frac * args.iters))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup, args.iters
    )

    @torch.no_grad()
    def run_val(n: int) -> float:
        model.eval()
        losses = []
        for i, batch in enumerate(valid_loader):
            if i >= n:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(out.loss.item())
        model.train()
        return sum(losses) / max(1, len(losses))

    print("starting training")
    model.train()
    t0 = time.perf_counter()
    train_iter = iter(train_loader)
    accum = 0
    optimizer.zero_grad()
    losses = []
    val0 = run_val(args.val_batches)
    print(f"Iter 0 (pre): Val loss {val0:.3f}")

    for step in range(1, args.iters + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / args.grad_accum
        loss.backward()
        accum += 1
        losses.append(out.loss.item())
        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum = 0

        if step % 10 == 0:
            mean_loss = sum(losses) / len(losses)
            losses = []
            elapsed = time.perf_counter() - t0
            it_per_s = step / elapsed
            peak = (torch.cuda.max_memory_allocated() / 1024**3
                    if torch.cuda.is_available() else 0)
            print(f"Iter {step}: Train loss {mean_loss:.3f}, "
                  f"LR {scheduler.get_last_lr()[0]:.2e}, "
                  f"It/sec {it_per_s:.3f}, "
                  f"Peak mem {peak:.2f} GB", flush=True)

        if step % args.val_every == 0 or step == args.iters:
            v = run_val(args.val_batches)
            print(f"Iter {step}: Val loss {v:.3f}", flush=True)

    print("saving adapter")
    model.save_pretrained(args.adapter_path)
    tokenizer.save_pretrained(args.adapter_path)

    print("merging adapter into base + saving fused HF checkpoint")
    del model
    torch.cuda.empty_cache()
    base, _, _ = load_base(args.model)
    fused = PeftModel.from_pretrained(base, args.adapter_path)
    fused = fused.merge_and_unload()
    Path(args.fused_path).mkdir(parents=True, exist_ok=True)
    # Saves the language_model subtree as a stand-alone causal-LM
    # checkpoint (model_type=gemma3_text for Gemma 3 4B, qwen3 for Qwen,
    # gemma3_text for Gemma 3 1B). convert_hf_to_gguf.py handles all three.
    fused.save_pretrained(args.fused_path, safe_serialization=True)
    tokenizer.save_pretrained(args.fused_path)

    print(f"done. fused-hf written to {args.fused_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
