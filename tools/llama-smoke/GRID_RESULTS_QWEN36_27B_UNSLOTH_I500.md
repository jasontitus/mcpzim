# llama.cpp grid — 2026-04-26 15:50

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✗ | 16449 | 555 | 555 | 59.8 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 16458 | 1057 | 1057 | 114.7 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 16466 | 901 | 901 | 97.7 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 16449 | 260 | 260 | 27.4 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 16451 | 412 | 411 | 44.0 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 16450 | 247 | 247 | 26.2 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 16445 | 266 | 266 | 28.3 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 16450 | 228 | 228 | 24.1 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 16447 | 210 | 210 | 22.3 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 16467 | 1363 | 1362 | 148.0 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 16468 | 1618 | 1617 | 174.6 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 16478 | 1913 | 1913 | 208.5 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 16468 | 1463 | 1462 | 159.4 |
