# llama.cpp grid — 2026-04-25 18:52

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✗ | 2822 | 0 | 0 | 11.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 2819 | 0 | 0 | 17.2 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 2816 | 0 | 0 | 6.0 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✗ | 2815 | 0 | 0 | 5.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 2815 | 0 | 0 | 5.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✗ | 2813 | 0 | 0 | 5.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 2814 | 0 | 0 | 5.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 2813 | 0 | 0 | 5.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 2813 | 0 | 0 | 5.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 2824 | 0 | 0 | 17.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 2823 | 0 | 0 | 17.5 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 2822 | 0 | 0 | 17.6 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 2820 | 0 | 0 | 17.9 |
