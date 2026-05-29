# llama.cpp grid — 2026-05-28 11:29

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5171 | 92 | 0 | 9.4 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✓ | 5175 | 200 | 0 | 20.7 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 5165 | 15 | 0 | 1.3 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5167 | 37 | 0 | 3.7 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 5169 | 76 | 0 | 7.6 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5168 | 33 | 0 | 3.3 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 5165 | 70 | 0 | 7.2 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 5166 | 28 | 0 | 2.8 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 5166 | 29 | 0 | 2.8 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5181 | 330 | 0 | 34.3 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5175 | 268 | 0 | 27.7 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5181 | 352 | 0 | 36.4 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 5177 | 335 | 0 | 34.7 |
