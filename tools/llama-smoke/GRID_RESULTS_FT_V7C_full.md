# llama.cpp grid — 2026-04-25 15:24

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| gemma3-4b-ft | Q4_K_M | f16/f16 | bars_sc_caltrain_chain | ✓ | 3236 | 0 | 0 | 3.9 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | sky_is_blue_chain | ✓ | 3264 | 0 | 0 | 3.1 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | restaurants_in_sf | ✓ | 3253 | 0 | 0 | 1.1 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | nearby_stories_palo_alto | ✓ | 3254 | 0 | 0 | 1.5 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | tell_me_about_palo_alto | ✓ | 3252 | 0 | 0 | 0.8 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | compare_musk_bezos | ✓ | 3250 | 0 | 0 | 1.2 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | relations_us_iran | ✗ | 3251 | 0 | 0 | 1.9 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | narrate_hp_garage | ✗ | 3251 | 0 | 0 | 1.1 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | what_is_here_in_sf | ✓ | 3252 | 0 | 0 | 1.1 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | grav_waves_chain | ✗ | 3264 | 0 | 0 | 2.5 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | wwi_vs_wwii_chain | ✗ | 3279 | 0 | 0 | 3.9 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | french_revolution_chain | ✓ | 3264 | 0 | 0 | 3.9 |
| gemma3-4b-ft | Q4_K_M | f16/f16 | crispr_chain | ✗ | 3268 | 0 | 0 | 3.2 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 3202 | 0 | 0 | 3.8 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✓ | 3193 | 0 | 0 | 3.5 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 3181 | 0 | 0 | 1.2 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 3183 | 0 | 0 | 1.6 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 3181 | 0 | 0 | 0.9 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 3184 | 0 | 0 | 1.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 3180 | 0 | 0 | 1.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 3177 | 0 | 0 | 1.3 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 3176 | 0 | 0 | 1.2 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 3192 | 0 | 0 | 2.8 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 3204 | 0 | 0 | 4.7 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✓ | 3193 | 0 | 0 | 4.4 |
| gemma3-4b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✓ | 3191 | 0 | 0 | 3.9 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | bars_sc_caltrain_chain | ✓ | 3162 | 0 | 0 | 4.3 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | sky_is_blue_chain | ✓ | 3149 | 0 | 0 | 3.3 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | restaurants_in_sf | ✓ | 3139 | 0 | 0 | 1.0 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | nearby_stories_palo_alto | ✓ | 3139 | 0 | 0 | 1.6 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | tell_me_about_palo_alto | ✓ | 3137 | 0 | 0 | 1.0 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | compare_musk_bezos | ✓ | 3142 | 0 | 0 | 1.3 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | relations_us_iran | ✓ | 3141 | 0 | 0 | 1.2 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | narrate_hp_garage | ✗ | 3139 | 0 | 0 | 1.1 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | what_is_here_in_sf | ✓ | 3141 | 0 | 0 | 1.2 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | grav_waves_chain | ✗ | 3154 | 0 | 0 | 3.1 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | wwi_vs_wwii_chain | ✗ | 3156 | 0 | 0 | 3.6 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | french_revolution_chain | ✓ | 3151 | 0 | 0 | 3.8 |
| gemma3-4b-ft | Q4_K_M | q4_0/q4_0 | crispr_chain | ✓ | 3149 | 0 | 0 | 3.6 |
