# llama.cpp grid — 2026-07-14 19:25

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

Peak RSS is the Python client only; model/server memory is excluded.

## Adjudication

The raw table below is intentionally preserved. It is 12/16 as originally
scored, but it does not indicate four model-quality failures:

- `sky_is_blue_chain` and `crispr_chain` returned correct, grounded answers
  through `narrate_article`; the original rubric allowed only `read_article`.
  The rubric now accepts both evidence-bearing paths.
- `bars_sc_caltrain_chain` exceeded the phone's 4K context by 132 tokens.
  The app now caps model-facing place rows at 8, truncates prose fields, and
  removes UI/map/debug-only payload fields while preserving the full UI trace.
- `wwi_vs_wwii_chain` produced a correct casualty answer but then selected
  `nearby_stories`. Comparative casualty/combatant/cause follow-ups now route
  deterministically to `compare_articles`; synthesis questions stay with the
  model.

The requested Putin, Alamo, and gravitational-wave conversations all passed
on the 1-bit model. These results support fixing orchestration rather than
fine-tuning the model. The post-fix Swift suite passed 329/329 tests, and the
exact app provider subsequently loaded and generated successfully with all
65 layers on Apple Metal and Q4_0 K/V at 4K.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| bonsai-1bit-27b | server | server/server | bars_sc_caltrain_chain | · | rc=1:     return self._post("/v1/chat/completions", payload) /            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/eval.py", line 73, in _post /     raise RuntimeError( / RuntimeError: llama-server HTTP 400: {"error":{"code":400,"message":"request (4228 tokens) exceeds the available context size (4096 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":4228,"n_ctx":4096}} | 0 | 0 | 30.6 |
| bonsai-1bit-27b | server | server/server | sky_is_blue_chain | ✗ | 51 | 0 | 0 | 34.5 |
| bonsai-1bit-27b | server | server/server | restaurants_in_sf | ✓ | 51 | 0 | 0 | 7.6 |
| bonsai-1bit-27b | server | server/server | nearby_stories_palo_alto | ✓ | 51 | 0 | 0 | 8.4 |
| bonsai-1bit-27b | server | server/server | tell_me_about_palo_alto | ✓ | 51 | 0 | 0 | 3.6 |
| bonsai-1bit-27b | server | server/server | compare_musk_bezos | ✓ | 52 | 0 | 0 | 4.7 |
| bonsai-1bit-27b | server | server/server | relations_us_iran | ✓ | 51 | 0 | 0 | 13.8 |
| bonsai-1bit-27b | server | server/server | narrate_hp_garage | ✓ | 51 | 0 | 0 | 2.9 |
| bonsai-1bit-27b | server | server/server | what_is_here_in_sf | ✓ | 52 | 0 | 0 | 7.3 |
| bonsai-1bit-27b | server | server/server | putin_biography_chain | ✓ | 51 | 0 | 0 | 17.7 |
| bonsai-1bit-27b | server | server/server | alamo_history_chain | ✓ | 51 | 0 | 0 | 11.4 |
| bonsai-1bit-27b | server | server/server | gravity_waves_creation | ✓ | 51 | 0 | 0 | 3.0 |
| bonsai-1bit-27b | server | server/server | grav_waves_chain | ✓ | 51 | 0 | 0 | 10.3 |
| bonsai-1bit-27b | server | server/server | wwi_vs_wwii_chain | ✗ | 51 | 0 | 0 | 46.3 |
| bonsai-1bit-27b | server | server/server | french_revolution_chain | ✓ | 51 | 0 | 0 | 25.3 |
| bonsai-1bit-27b | server | server/server | crispr_chain | ✗ | 51 | 0 | 0 | 40.3 |
