# GSQ-RCO → Apple Silicon: session handoff

**Date**: 2026-09-20
**Project root**: `/Users/jasontitus/experiments/mcpzim` (this is the repo — an
earlier session ran from `/Users/jasontitus/experiments/gsq-rco-mlx`, which is an
empty clone and the WRONG tree; everything below lives under `mcpzim`)
**Solver**: `tools/calibration/solver/`
**Log of record**: `docs/PORT_VALIDATION.md`

Run shell commands from `tools/calibration` with `PYTHONPATH=.` and the venv at
`tools/calibration/.venv/bin/python`.

---

## 0. STATUS UPDATE — 2026-09-20, later session (read before §2/§3)

§1 is still accurate. §2's blocker, §3's unfinished list, §3.4's disk arithmetic and
§5's single-block command are not: this section supersedes them.

### Fixed, each with a reproduction

1. **The per-block driver could not run.** The heredoc that was meant to give each
   iteration a fresh `attempt_id`, `output`, `gsq_max_blocks=1` and
   `resume_checkpoint` built all four in memory and never wrote them back, so every
   iteration ran the stale config on disk — iteration 1 dies immediately on the
   existing `output` directory. §3.1's "compiled but unverified" understated it: the
   write was absent. Rewritten, and it now **completes two blocks** (see below).

2. **A resumed process discarded every previously-trained block.** `gsq_run` dropped
   each completed block's archive from `extras`, and the block anchor filtered
   `candidate_block_*` out of its payload, so on resume every earlier block fell back
   to its untrained RTN `initial_candidates` while the run reported `completed`.
   Reproduced on a 2-layer CPU model (8 of layer 0's modules differed from a
   single-process reference; now 0 differ) and confirmed on the real model: the
   anchor iteration 1 published carries `candidate_block_0`, where the pre-fix anchor
   in `checkpoints/gsq/20260920-075700-ba75` carries none. The read side is now
   guarded too — a snapshot missing a completed block's archive is refused by name,
   because `initial_candidates` otherwise supplies a valid candidate for every module
   NAME and the loss is invisible.

3. **`checkpointed_stop` was read as "all blocks complete"** (`next_block` absent →
   `-1` → exit 0). That is what a deadline or SIGINT produces; `gsq-anchor1`'s report
   is exactly that shape. The driver now branches on `status`, resumes a mid-block
   stop without advancing the block counter, and refuses to start past `deadline_unix`.

4. **The zero-step guard rejected a legitimate resume** from a block's own end state,
   which would have wedged a sweep whose stop landed on a block's final update.
   Exempted for a block already complete on entry; a fresh block with zero steps still
   raises.

5. **Disk: §3.4 understates it by ~2.3x.** Per-attempt store roots re-store the
   invariants (16.6 GB/attempt measured). Live: block 0 = 26.1 GiB of store + 6.1 GiB
   of output; a resumed attempt = ~23 GiB of store + 29 GiB of output, of which
   `restored-<snapshot>` is 16.61 GB. ~52 GiB/block ⇒ exhaustion near **block 6**
   against 511 GiB free, not block 25. The driver now reclaims `restored-*`; the two
   store levers still need a decision. See `docs/PORT_VALIDATION.md`.

6. **The tree arrived with two failing tests.** Now **529 passed, 3 skipped**
   (`cd tools/calibration && PYTHONPATH=. .venv/bin/python -m pytest tests/ solver/tests/ -q`),
   including regression tests for defects 2 and 4.

### The §3.1 dry run, done

`./runs/native-mps/run-gsq-per-block.sh 2`, real 52 GiB model, 25.2 min:

```
block 0 (fresh)  -> blocked_stop snapshot=gsq-b001-s00000087-e000-q00000-p00002 next_block=1
block 1 (resume) -> blocked_stop snapshot=gsq-b002-s00000174-e000-q00000-p00002 next_block=2
reached the block limit (2)
```

`candidate_block_0`'s sha256 is identical in iteration 1's anchor and iteration 2's,
so the trained block survived the process boundary byte-exact.

### Still open

- **The block-3 root cause is still not found.** The workaround stands. The block
  *entry* state is bit-identical across six independent attempts
  (solver/optimizer/rng/scheduler byte-equal), but `cache_block_1` is NOT invariant
  across those attempts, so the trained-block equivalence rests on the anchor
  semantics plus the CPU fixture — not on a real-model measurement.
- **The disk levers need a decision** before a 64-block sweep: reclaim
  `restored-<snapshot>` (done in the driver), one shared store root with
  attempt-scoped snapshot names (~1 TB), delete the abandoned attempt roots (288 GB,
  which are also the evidence for several comparisons in `docs/PORT_VALIDATION.md`).
- **`runtime_sha256` is stale and nothing checks it.** It is computed only by
  `solver/tools/make_run_configs.py`, so this session's edits to `solver/run.py` and
  `checkpoints.py` mean the config's pin no longer describes the tree while every
  receipt still asserts it. Regenerate before a production sweep, and expect the new
  pin to invalidate the existing anchor chain (identity is what `restore` verifies).
- **`tools/calibration` is untracked in git** — 3.8k lines of solver and tests, never
  committed, with 377 GB of stores beside it. `git clean -xfd` removes the port. Its
  own `.gitignore` already excludes `runs/`, so code, tests and docs can be committed.
- A second copy of the port at
  `tools/calibration/cuda_runtime/.context/application/solver/` has already diverged
  and is generated by no script in the tree.

### §5 corrections

The single-block command in §5 fails as written: the config's `output` names an
existing directory (`runs/native-mps/gsq-anchor1`) and its `attempt_id` collides with
an existing store root — the exact trap §5's own gotchas list. Both keys are excluded
from the checkpoint identity, so refreshing them is safe. Prefer the driver
(`./runs/native-mps/run-gsq-per-block.sh N`).

## 1. What this is

Porting the GSQ (Gumbel-Softmax Quantization) post-training quantizer — upstream
NVIDIA/CUDA, `IST-DASLab/GSQ` — to Apple Silicon so Qwen3.8-27B can be quantized
on an M1/M5. Upstream is staged under
`tools/calibration/cuda_runtime/.context/sources/{gsq,rco}`.

Target model: Qwen3.8-27B, 64 layers, hybrid `linear_attention` + `full_attention`,
52 GiB at bf16. Calibration corpus: 87 invocations, 104,557 tokens.

---

## 2. Status: broken at block 3, one documented workaround

### Proven good

- **Capture parity** — `runs/native-mps/parity-report.txt`, exit 0.
  `first_block_input` **bit-exact** (0 ulp); `final_normalized_hidden` 0.80 ulp
  mean against a 3.0 ulp tolerance. This validates weight loading, dtype,
  attention, rotary, and hybrid dispatch against tensors produced by a
  *different runtime* (the MLX capture).
- **Real GSQ training on MPS** — blocks 0, 1 and 2 each complete with real
  optimizer steps and exported candidates. One block ≈ 11 min including a
  ~50 s checkpoint publish. Losses descend (block 0 → 0.034).
- **Tests** — 349 passed, 3 skipped (the 3 need the external `manifold` package).
  Four new tests in `tests/test_execution_policy.py` fail on the pre-fix code.

### The blocker

**Block 3** (the first `full_attention` block) fails reproducibly with
`Nonfinite/missing GSQ gradient` after ~261 accumulated steps in one process.
Blocks 0–2 are `linear_attention` and always succeed.

Established by measurement — it is **none** of these:

| hypothesis | verdict |
|---|---|
| broken maths | **no** — fresh process trains block 3 for 87/87 steps, loss 509→38 |
| learning rate | **no** — gradient norm 93/92/86 at lr 1e-3/1e-4/1e-5, all finite |
| warmstart | **no** — passes with and without; scales 0.02–0.34, well inside FP16 |
| corrupt cache | **no** — student \|max\| 1456 vs 1448, teacher 75.0, all finite |
| store collision | **no** (though that was real and is separately fixed) |
| my own edits | **no** — they touch `extras`, the store root, policy device, digest |
| sequence-replay of blocks 0-2 then 3 in one process | **no** — still passes |

**Unreproducible factor**: ~261 prior steps of MPS process state. Free memory
measured as low as **4.0 GB** against 46.3 of 47.1 GB swap in use.

### The workaround (INCOMPLETE — see §3)

Run one block per OS process: `runs/native-mps/run-gsq-per-block.sh`.
Block boundaries are exact resume points, and a fresh process trains block 3
fine, so this routes around the failure. Cost ≈ 25 s of model load per block.

---

## 3. Unfinished work — read before continuing

1. **The per-block driver has never completed two blocks.**
   `run-gsq-per-block.sh` (and the `gsq_max_blocks` support in `gsq_run`) was
   written but only partly exercised. A stop placed *before* the block loop
   committed only `b<N>` and left no `b<N+1>` anchor, so `--resume` pointed at a
   nonexistent snapshot. That has just been rewritten to stop *after* the block
   and publish the next block's phase-start anchor first — **this fix is
   compiled but unverified.** Verify with a 2-block dry run before trusting it.

2. **Two adversarial reviews were running when the session ended.** They were
   initially pointed at the wrong repo (see §6). Check:
   `hub` job ids `DriverReview` and `ResumeStateReview`, or
   `agent://DriverReview` / `agent://ResumeStateReview`. The open question they
   were asked to settle: **does a per-block restart change the trained result
   versus one long process** (RNG stream and LR schedule continuity)? If it does,
   that must be disclosed in `docs/PORT_VALIDATION.md`.

3. **The block-3 root cause is not found.** The workaround is a mitigation, not a
   fix. The likely mechanism is MPS returning bad values under host memory
   pressure on a unified-memory machine, but that is **[INFERENCE]**, not proven.

4. **Checkpoint store projects to ~693 GB** over 64 blocks (`solver`+`optimizer`
   change every commit at 4.32 GB). 902 GB free, so it fits with ~200 GB margin.
   The lever is `checkpoint_seconds` (currently 1200).

---

## 4. Bugs found and fixed this session

In `docs/PORT_VALIDATION.md` with full evidence; summary:

1. **`torch.device('mps') != torch.device('mps', 0)`** — rejected correctly
   resident baselines. Identical trap on CUDA (`cuda` vs `cuda:0`), only avoided
   there because the NVIDIA configs specify indices. Fix:
   `device_policy.same_device(a, b)`.
2. **`full` residency killed the machine** — 50.1 GiB on the accelerator, silent
   external kill, swap exhausted. `block_cpu_offload` now measures **0.71 GiB
   peak**, 70× less. This is the mode the config uses.
3. **Checkpoint store collision** — same snapshot name, different bytes, dies
   mid-run looking like a numerical failure. It masked a *passing* result across
   three attempts. Fixed by scoping the write root per `attempt_id`.
4. **Policy could report `backend: cuda` for an MPS run** — three independent
   device resolutions. Now one, resolved before policy and gate.
5. **`exact_rng_resume: False` contradicted a passing test** — split into
   `exact_rng_resume` (true) and `exact_arithmetic_replay` (false on MPS).
6. **Non-atomic cache writes** — a truncated `.pt` would be skipped forever by an
   `exists()` check. Now pending+`os.replace`.
7. **Runtime digest missed checkpoint-format changes** — now covers
   `checkpoints.py`.

### A guard added, measured, and REMOVED (important lesson)

A skip-and-continue guard for non-finite gradients carried the run to block 5
while **every step was skipped**, so block 3 exported its warmstart back out
byte-identical and was recorded as *calibrated*. Silent corruption from a guard
meant to add robustness. Removed; a non-finite gradient now fails immediately with
the block and sequence named. Replaced by two checks in `gsq_run`:
`steps_in_block==0` raises, and every export must `torch.equal` its quantizer's
hard weights.

---

## 5. How to run it

```bash
cd /Users/jasontitus/experiments/mcpzim/tools/calibration

# tests
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q

# one block, bounded
PYTHONPATH=. .venv/bin/python -m solver.run \
    --config runs/native-mps/gsq-run.json --stage gsq

# the full sweep (verify §3.1 first)
./runs/native-mps/run-gsq-per-block.sh 64
```

Config: `runs/native-mps/gsq-run.json`. Key fields: `gsq_memory_mode:
block_cpu_offload`, `gsq_execution_device: mps`, `attempt_id` (fresh per
attempt), `output` (must be empty/nonexistent), `checkpoint_seconds: 1200`,
`gsq_max_blocks` (1 for per-block), `deadline_unix`.

**Gotchas that cost time this session:**
- An existing non-empty `output` dir raises `FileExistsError`. That is by design.
- Never `rm -rf` an output/store directory while a run is using it. I did this
  three times and each time produced a confusing error that looked like a code
  bug (the stale log from one such incident then misled a reviewer).
- `warmstart_states` are REAL inputs at `runs/native-mps/smoke/warmstart-block-{0,3}.pt`.
  Do not delete them as scratch — that killed a run.
- Reusing an `attempt_id` collides in the store. Use a fresh one.

---

## 6. Session-level note

This session's cwd was `/Users/jasontitus/experiments/gsq-rco-mlx`, which holds
**no code** — an artifact of an early `git clone` that went to the wrong path.
Every path above is absolute into `mcpzim`. Two subagent reviews were misdirected
by this before being corrected. **Start the next session from
`/Users/jasontitus/experiments/mcpzim`** so LSP, greps and subagents resolve
correctly.