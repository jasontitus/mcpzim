# llama.cpp grid — 2026-04-26 02:57

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 2975 | 0 | 0 | 26.8 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 3009 | 0 | 0 | 20.6 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 3004 | 0 | 0 | 8.8 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 2997 | 0 | 0 | 9.3 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 3003 | 0 | 0 | 8.8 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 3000 | 0 | 0 | 8.8 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 3008 | 0 | 0 | 7.7 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 3001 | 0 | 0 | 8.3 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 2999 | 0 | 0 | 3.5 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 3021 | 0 | 0 | 19.4 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 3024 | 0 | 0 | 34.0 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✓ | 3011 | 0 | 0 | 24.5 |
| qwen3.5-4b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 3029 | 0 | 0 | 28.6 |
