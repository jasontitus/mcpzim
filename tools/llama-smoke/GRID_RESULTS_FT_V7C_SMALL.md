# llama.cpp grid — 2026-04-25 22:48

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 996 | 0 | 0 | 2.0 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 989 | 0 | 0 | 1.5 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 980 | 0 | 0 | 0.3 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✗ | 983 | 0 | 0 | 0.7 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 990 | 0 | 0 | 1.0 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✗ | 981 | 0 | 0 | 0.4 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 980 | 0 | 0 | 1.7 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 982 | 0 | 0 | 0.6 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 981 | 0 | 0 | 0.2 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 987 | 0 | 0 | 1.9 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 986 | 0 | 0 | 7.9 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 986 | 0 | 0 | 2.0 |
| gemma3-1b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 987 | 0 | 0 | 4.8 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 3181 | 0 | 0 | 5.5 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 3169 | 0 | 0 | 13.1 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 3168 | 0 | 0 | 1.3 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 3174 | 0 | 0 | 1.8 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 3169 | 0 | 0 | 2.1 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 3170 | 0 | 0 | 1.3 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 3170 | 0 | 0 | 1.5 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 3170 | 0 | 0 | 1.3 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 3167 | 0 | 0 | 1.9 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 3177 | 0 | 0 | 2.7 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 3175 | 0 | 0 | 3.2 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 3176 | 0 | 0 | 5.0 |
| qwen3-4b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 3172 | 0 | 0 | 6.8 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 1717 | 0 | 0 | 2.2 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 1712 | 0 | 0 | 1.8 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 1701 | 0 | 0 | 0.3 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✗ | 1702 | 0 | 0 | 0.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 1704 | 0 | 0 | 0.7 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 1703 | 0 | 0 | 0.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 1705 | 0 | 0 | 0.7 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 1703 | 0 | 0 | 0.6 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 1702 | 0 | 0 | 0.7 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 1710 | 0 | 0 | 1.2 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 1714 | 0 | 0 | 1.8 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 1712 | 0 | 0 | 1.2 |
| qwen3-1.7b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 1713 | 0 | 0 | 1.3 |
