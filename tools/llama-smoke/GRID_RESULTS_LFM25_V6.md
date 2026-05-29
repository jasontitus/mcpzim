# llama.cpp grid — 2026-05-28 22:29

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5093 | 25 | 0 | 2.5 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5164 | 18 | 0 | 1.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 5158 | 9 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5158 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5157 | 8 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5158 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 5159 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 5162 | 8 | 0 | 0.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 5157 | 9 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✓ | 5163 | 21 | 0 | 2.2 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✓ | 5164 | 20 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5162 | 19 | 0 | 2.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✓ | 5165 | 26 | 0 | 2.7 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5102 | 54 | 0 | 5.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5181 | 231 | 0 | 24.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 5162 | 25 | 0 | 2.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5161 | 31 | 0 | 3.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5159 | 17 | 0 | 1.7 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5160 | 28 | 0 | 2.8 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 5164 | 77 | 0 | 7.9 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 5163 | 16 | 0 | 1.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 5159 | 12 | 0 | 1.2 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5178 | 280 | 0 | 29.3 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5178 | 232 | 0 | 24.0 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5177 | 234 | 0 | 24.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 5171 | 113 | 0 | 11.6 |
