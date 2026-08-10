# llama.cpp grid — 2026-04-26 02:39

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5599 | 217 | 0 | 22.4 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5597 | 226 | 0 | 24.3 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 5591 | 114 | 0 | 12.4 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5589 | 59 | 0 | 6.4 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5588 | 42 | 0 | 4.5 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5588 | 47 | 0 | 5.0 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 5590 | 51 | 0 | 5.5 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 5588 | 52 | 0 | 5.6 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 5590 | 30 | 0 | 3.2 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5595 | 197 | 0 | 21.3 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✓ | 5600 | 340 | 0 | 36.9 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5595 | 223 | 0 | 24.3 |
| qwen3-8b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✓ | 5594 | 257 | 0 | 28.0 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✗ | 5814 | 436 | 0 | 47.1 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5794 | 141 | 0 | 15.1 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 5797 | 239 | 0 | 25.8 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5789 | 73 | 0 | 7.7 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 5795 | 122 | 0 | 13.1 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5788 | 63 | 0 | 6.6 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 5790 | 116 | 0 | 12.5 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 5784 | 73 | 0 | 7.8 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 5788 | 43 | 0 | 4.5 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5813 | 337 | 0 | 36.3 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5811 | 335 | 0 | 36.0 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5815 | 386 | 0 | 41.9 |
| qwen3.5-9b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 5795 | 213 | 0 | 23.1 |
