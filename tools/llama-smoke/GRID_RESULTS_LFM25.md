# llama.cpp grid — 2026-05-28 11:24

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5172 | 77 | 0 | 7.8 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5170 | 136 | 0 | 14.0 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 5167 | 69 | 0 | 7.0 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5167 | 39 | 0 | 3.9 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 5167 | 86 | 0 | 8.8 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✗ | 5163 | 38 | 0 | 3.9 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 5162 | 39 | 0 | 3.9 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 5168 | 79 | 0 | 8.0 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 5165 | 41 | 0 | 4.1 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5174 | 247 | 0 | 25.6 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5169 | 197 | 0 | 20.4 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5168 | 63 | 0 | 6.3 |
| lfm2.5-8b-a1b | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 5162 | 76 | 0 | 7.8 |
