"""LoRA fine-tune via Unsloth's FastLanguageModel (4-bit base + 16-bit
LoRA + their patched grad-checkpointing). Drop-in alternative to
finetune_cuda.py for Qwen 3.5 / 3.6 where vanilla PEFT QLoRA hits
quality regressions and bf16 LoRA doesn't fit on a 24 GB card.

Same CLI shape as finetune_cuda.py so finetune_unsloth.sh can wrap
this in the same split → train → fuse → convert → quantize chain.

After training, the adapter saves in standard PEFT format. The fuse
step still happens in pure transformers (load bf16 base, attach
adapter, merge_and_unload, save fused-hf) — Unsloth's 4-bit weights
can't be merged in place.
"""
from __future__ import annotations

# IMPORTANT: import unsloth BEFORE transformers / peft, per their docs.
# Unsloth monkey-patches a bunch of things at import time and the patches
# are silently lost if transformers loads first.
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_row(tokenizer, row: dict, max_seq_length: int,
               enable_thinking: bool) -> dict:
    """Apply chat template + return as a {'text': str} record so SFTTrainer
    can consume it via its built-in tokenization. enable_thinking=False
    forces the assistant turn to skip the open <think>\\n marker, so the
    model learns to answer directly — fixes the Qwen 3.5/3.6 looping
    issue we saw on the eval grid.
    """
    messages = row["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return {"text": text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF repo (e.g. Qwen/Qwen3.6-27B)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--fused-path", required=True,
                    help="(unused here; fuse happens in finetune_unsloth.sh)")
    ap.add_argument("--iters", type=int, default=500,
                    help="max_steps for SFTTrainer")
    ap.add_argument("--num-layers", type=int, default=16,
                    help="ignored (Unsloth applies LoRA across all layers); "
                         "kept for shell-script compat")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--val-batches", type=int, default=5,
                    help="(unused; SFTTrainer doesn't easily expose mid-run "
                         "val without an eval dataset, and we skip-eval here)")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--load-in-4bit", action="store_true", default=True,
                    help="Default-on; pass --no-load-in-4bit for bf16 LoRA "
                         "(needs ~56 GB on 27B, won't fit a 24 GB card)")
    ap.add_argument("--no-load-in-4bit", dest="load_in_4bit",
                    action="store_false")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Train with the open <think>\\n marker. Default off "
                         "— Qwen 3.5/3.6 with thinking-mode loops on our eval "
                         "grid without producing a final response.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"loading {args.model} (4bit={args.load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
        dtype=torch.bfloat16,
    )

    print(f"applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable/1e6:.2f}M / {total/1e6:.0f}M "
          f"({100*trainable/total:.3f}%)")

    print(f"loading + tokenizing data (enable_thinking={args.enable_thinking})")
    data_dir = Path(args.data_dir)
    train_rows = load_jsonl(data_dir / "train.jsonl")
    valid_rows = load_jsonl(data_dir / "valid.jsonl")
    print(f"  train={len(train_rows)} valid={len(valid_rows)}")

    train_ds = Dataset.from_list(train_rows).map(
        lambda r: render_row(tokenizer, r, args.max_seq_length,
                             args.enable_thinking),
        remove_columns=["messages"],
        desc="train",
    )
    valid_ds = Dataset.from_list(valid_rows).map(
        lambda r: render_row(tokenizer, r, args.max_seq_length,
                             args.enable_thinking),
        remove_columns=["messages"],
        desc="valid",
    )

    # IMPORTANT: eval_strategy="no". Mid-run eval on a 27B at the
    # 24 GB-3090 ceiling pushes Unsloth to enable "smartly offload
    # gradients to save VRAM" — saves memory but cuts throughput
    # ~5-7×. Once on, it stays on for the rest of training. We
    # skip mid-run eval and rely on grid eval against saved
    # checkpoints. The valid set is still used by SFTTrainer for
    # final-epoch metrics if it falls within max_steps.
    sft_cfg = SFTConfig(
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=max(1, int(0.05 * args.iters)),
        max_steps=args.iters,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_every,
        output_dir=f"{args.adapter_path}-trainer",
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=args.seed,
        report_to="none",
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=sft_cfg,
    )
    trainer.train()

    print(f"saving adapter to {args.adapter_path}")
    Path(args.adapter_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_path)
    tokenizer.save_pretrained(args.adapter_path)

    # --- fuse into bf16 base ---
    # Unsloth's 4-bit weights can't be merged in place. Drop the trained
    # 4-bit model, free the GPU, reload the base in bf16 via vanilla
    # transformers, attach the just-saved adapter, merge, save fused-hf.
    print("merging adapter into bf16 base + saving fused HF checkpoint")
    del model
    del trainer
    torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    fused = PeftModel.from_pretrained(base, args.adapter_path)
    fused = fused.merge_and_unload()
    Path(args.fused_path).mkdir(parents=True, exist_ok=True)
    fused.save_pretrained(args.fused_path, safe_serialization=True)
    base_tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    base_tok.save_pretrained(args.fused_path)
    print(f"done. fused-hf written to {args.fused_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
