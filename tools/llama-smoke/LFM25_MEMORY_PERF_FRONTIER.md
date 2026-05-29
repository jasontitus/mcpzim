# LFM2.5-8B-A1B FT — memory × accuracy frontier (2026-05-29)

Goal: make the fine-tuned LFM2.5 (a) use less RAM than its 5.16 GB
Q4_K_M baseline and (b) beat Gemma 3 4B FT V7C (10/13 @ 3.2 GB RSS).
**Both achieved by v7-full Q3_K_M (12/13 @ 4.16 GB).**

Eval = 13-scenario tool-call grid (`grid.py`), peak RSS measured on
M2 Max via `eval.py`. KV = q8_0/q8_0 unless noted.

## Frontier (q8_0/q8_0 KV)

| model / quant              | passes /13 | avg peak RSS | vs Gemma |
|----------------------------|-----------|--------------|----------|
| **v7-full Q3_K_M** ★       | **12/13** | **4.16 GB**  | **+2 acc, +1 GB RAM** |
| v7-full Q4_K_M             | 11/13     | 5.16 GB      | +1 acc, +2 GB |
| v7 iter-200 Q3_K_M         | 10/13     | 4.16 GB      | ties, +1 GB |
| v7 iter-200 Q4_K_M         | 10/13     | 5.16 GB      | ties, +2 GB |
| v6 Q4_K_M                  | 10/13     | 5.16 GB      | ties |
| v6 Q3_K_M                  | 8/13      | 4.16 GB      | −2 |
| v7/v6 Q2_K                 | 3/13      | 3.28 GB      | collapses |
| Gemma 3 4B FT V7C (ref)    | 10/13     | 3.18 GB      | baseline |

q4_0 KV (vs q8_0) cost 1-2 scenarios for ~30 MB — not worth it; use q8_0.
Q2_K reaches Gemma's RAM (3.28 GB) but collapses to 3/13 — too lossy for
this MoE.

## RESULT: goal met

**Shippable winner = v7-full Q3_K_M @ q8_0 KV: 12/13 @ 4.16 GB.**
- **Accuracy: 12/13 BEATS Gemma's 10/13 by 2 scenarios.**
- **Memory: 4.16 GB, down 1.0 GB (−19%) from the 5.16 GB Q4_K_M baseline.**
- Decode ~1.8× faster than Gemma 3 4B FT (1.5B active MoE vs 4B dense),
  46 tok/s on iPhone via Enclave; multilingual.
- Still ~1 GB heavier than Gemma's 3.18 GB (the 8.3B MoE keeps all
  experts resident; Q2_K is the only quant that reaches Gemma's RAM and
  it destroys accuracy). So: beats Gemma on accuracy + speed, modestly
  heavier on RAM, far lighter than the LFM2.5 Q4_K_M baseline.

v7-full Q4_K_M (11/13 @ 5.16 GB) also beats Gemma but Q3_K_M dominates
it (higher accuracy, less RAM). The Q3 vs Q4 difference is quant noise on
the two borderline chains: Q4 passes grav_waves but fails wwi+french;
Q3 fails grav_waves but passes wwi+french → Q3 nets 12 vs Q4's 11.

## What made v7 win (vs v6 / v7-iter200, both 10/13)

The 317 targeted hard-case rows ([[generate-chains-3-turn]] explainer +
narrate templates) + FULL 800-iter training (not the iter-200 early-stop):
- `sky_is_blue_chain`: ✗ (v6) → ✓ — explainer template
- `narrate_hp_garage`: ✗ (every prior run) → ✓ — narrate template +
  enough iters to learn the new `narrate_article` tool
- `grav_waves_chain`: ✓ at Q4 (full training recovered what iter-200 lost)

The iter-200 early-stop (done to hit a deadline) was the only thing that
had held v7 at 10/13; full training cleared the bar.

## Remaining fails (12/13)

v7-full Q3_K_M fails only `grav_waves_chain` (passes at Q4_K_M, so it's a
quant-borderline 3-turn chain). v7-full Q4_K_M fails `wwi_vs_wwii_chain`
+ `french_revolution_chain`. No single config is 13/13; the borderline
chains flip with quant.

## Artifacts

- **v7-full (winner): `tools/fine-tune/ft-out-lfm2.5-8b-v7full/`** —
  Q3_K_M 3.83 GB (12/13) + Q4_K_M 4.80 GB (11/13) + adapters.
- v7 iter-200: `ft-out-lfm2.5-8b-v7iter200/` (10/13).
- v6: `ft-out-lfm2.5-8b-v6/` (10/13).
- Result grids: `GRID_RESULTS_LFM25_V7FULL.md` (Q4_K_M 11/13),
  `/tmp/v7full_q3km.md` (Q3_K_M 12/13), `GRID_RESULTS_LFM25_MEM.md`
  (full memory sweep).
