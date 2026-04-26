# llama.cpp grid — 2026-04-26 11:44

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✗ | 16449 | 571 | 571 | 59.2 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 16460 | 958 | 958 | 99.6 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 16470 | 1319 | 1318 | 137.5 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 16447 | 416 | 416 | 43.0 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 16449 | 429 | 429 | 45.2 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 16446 | 470 | 470 | 48.6 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 16452 | 730 | 730 | 75.8 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 16446 | 221 | 221 | 22.4 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 16446 | 229 | 229 | 23.4 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 16464 | 1163 | 1163 | 121.4 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 16452 | 859 | 859 | 91.3 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 16468 | 1370 | 1369 | 149.2 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✓ | 16461 | 1249 | 1249 | 135.9 |
