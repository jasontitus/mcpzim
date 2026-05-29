# llama.cpp grid — 2026-05-28 13:18

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5098 | 27 | 0 | 2.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5163 | 22 | 0 | 2.2 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 5157 | 10 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✗ | 5157 | 11 | 0 | 1.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5160 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5160 | 10 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 5157 | 13 | 0 | 1.3 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 5156 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 5157 | 7 | 0 | 0.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5165 | 21 | 0 | 2.2 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5163 | 21 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✓ | 5163 | 21 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✓ | 5163 | 24 | 0 | 2.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5170 | 45 | 0 | 4.7 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5179 | 194 | 0 | 20.2 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 5162 | 21 | 0 | 2.2 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5159 | 25 | 0 | 2.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5158 | 15 | 0 | 1.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5159 | 24 | 0 | 2.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 5163 | 66 | 0 | 6.8 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 5159 | 15 | 0 | 1.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 5158 | 12 | 0 | 1.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5179 | 241 | 0 | 25.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5179 | 209 | 0 | 21.8 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5178 | 214 | 0 | 22.3 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 5170 | 104 | 0 | 10.8 |
