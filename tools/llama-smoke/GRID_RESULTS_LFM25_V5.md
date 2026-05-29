# llama.cpp grid — 2026-05-28 20:00

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5163 | 26 | 0 | 2.3 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5162 | 17 | 0 | 1.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 5156 | 9 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5157 | 9 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5156 | 8 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5156 | 9 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 5156 | 8 | 0 | 0.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 5156 | 8 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 5156 | 7 | 0 | 0.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5164 | 17 | 0 | 1.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5160 | 18 | 0 | 1.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5162 | 15 | 0 | 1.6 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✓ | 5162 | 25 | 0 | 2.6 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5100 | 39 | 0 | 4.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5178 | 193 | 0 | 20.7 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 5162 | 20 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5159 | 26 | 0 | 2.7 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5159 | 13 | 0 | 1.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5159 | 23 | 0 | 2.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 5165 | 62 | 0 | 6.7 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 5158 | 13 | 0 | 1.3 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 5158 | 11 | 0 | 1.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5178 | 219 | 0 | 23.9 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5179 | 180 | 0 | 19.7 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5177 | 185 | 0 | 20.2 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 5170 | 90 | 0 | 9.7 |
