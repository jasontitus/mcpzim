# llama.cpp grid — 2026-04-25 01:19

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 3210 | 0 | 0 | 5.1 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✓ | 3199 | 0 | 0 | 3.5 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 3182 | 0 | 0 | 1.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✗ | 3192 | 0 | 0 | 2.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 3181 | 0 | 0 | 1.2 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 3181 | 0 | 0 | 1.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 3181 | 0 | 0 | 2.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 3178 | 0 | 0 | 1.0 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 3178 | 0 | 0 | 1.1 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 3193 | 0 | 0 | 2.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 3204 | 0 | 0 | 5.5 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 3195 | 0 | 0 | 3.1 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 3190 | 0 | 0 | 3.8 |
