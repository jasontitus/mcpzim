# llama.cpp grid — 2026-04-25 15:53

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 1719 | 0 | 0 | 2.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 1713 | 0 | 0 | 1.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 1707 | 0 | 0 | 0.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✗ | 1705 | 0 | 0 | 0.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 1704 | 0 | 0 | 0.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 1707 | 0 | 0 | 0.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 1705 | 0 | 0 | 0.7 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 1702 | 0 | 0 | 0.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 1702 | 0 | 0 | 0.4 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 1711 | 0 | 0 | 1.5 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 1710 | 0 | 0 | 1.5 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 1712 | 0 | 0 | 1.4 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 1710 | 0 | 0 | 1.2 |
