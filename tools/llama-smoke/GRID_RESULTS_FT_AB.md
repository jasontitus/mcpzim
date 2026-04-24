# llama.cpp grid — 2026-04-24 08:12

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| gemma3-4b | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 2792 | 0 | 0 | 3.9 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 2836 | 0 | 0 | 6.4 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 2822 | 0 | 0 | 1.5 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✗ | 2818 | 0 | 0 | 1.7 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 2827 | 0 | 0 | 1.9 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 2821 | 0 | 0 | 1.7 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 2821 | 0 | 0 | 1.9 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 2822 | 0 | 0 | 2.3 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 2821 | 0 | 0 | 1.8 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 2834 | 0 | 0 | 4.5 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 2835 | 0 | 0 | 6.4 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 2837 | 0 | 0 | 4.8 |
| gemma3-4b | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 2834 | 0 | 0 | 5.2 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✗ | 3137 | 0 | 0 | 3.6 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 3180 | 0 | 0 | 8.0 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 3188 | 0 | 0 | 5.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 3178 | 0 | 0 | 3.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 3176 | 0 | 0 | 1.6 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 3177 | 0 | 0 | 3.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 3178 | 0 | 0 | 3.1 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 3177 | 0 | 0 | 2.2 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 3176 | 0 | 0 | 2.2 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 3183 | 0 | 0 | 3.7 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 3189 | 0 | 0 | 12.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 3190 | 0 | 0 | 9.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 3179 | 0 | 0 | 10.6 |
