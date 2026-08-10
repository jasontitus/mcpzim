# llama.cpp grid — 2026-04-26 02:20

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✗ | 16202 | 134 | 131 | 10.2 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 16453 | 128 | 127 | 13.4 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 16445 | 45 | 44 | 4.2 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✗ | 16447 | 75 | 74 | 7.4 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 16453 | 92 | 91 | 9.1 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✗ | 16445 | 38 | 37 | 3.2 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 16447 | 140 | 139 | 14.3 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 16447 | 100 | 100 | 10.0 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 16451 | 33 | 32 | 2.7 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 16454 | 151 | 150 | 15.5 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 16455 | 263 | 262 | 27.8 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 16450 | 210 | 209 | 22.2 |
| qwen3.6-27b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 16454 | 184 | 183 | 19.2 |
