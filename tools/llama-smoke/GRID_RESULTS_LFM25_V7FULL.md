# llama.cpp grid — 2026-05-29 08:31

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5097 | 28 | 0 | 2.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✓ | 5162 | 20 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 5157 | 10 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5159 | 11 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5157 | 8 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5157 | 12 | 0 | 1.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 5157 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 5157 | 9 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 5156 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✓ | 5161 | 23 | 0 | 2.4 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5162 | 19 | 0 | 1.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5162 | 17 | 0 | 1.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✓ | 5162 | 23 | 0 | 2.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5101 | 40 | 0 | 4.2 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5178 | 192 | 0 | 20.6 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 5162 | 20 | 0 | 2.0 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5159 | 26 | 0 | 2.7 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5159 | 15 | 0 | 1.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5159 | 23 | 0 | 2.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 5163 | 65 | 0 | 7.0 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 5159 | 14 | 0 | 1.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 5159 | 11 | 0 | 1.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5178 | 240 | 0 | 25.9 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5178 | 191 | 0 | 20.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5176 | 198 | 0 | 21.2 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 5169 | 97 | 0 | 10.4 |
