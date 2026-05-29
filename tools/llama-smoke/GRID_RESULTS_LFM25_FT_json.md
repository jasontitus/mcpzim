# llama.cpp grid — 2026-05-28 12:16

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5173 | 25 | 0 | 2.6 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5164 | 25 | 0 | 2.6 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 5157 | 10 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5157 | 12 | 0 | 1.2 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5156 | 10 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✗ | 5155 | 6 | 0 | 0.5 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 5156 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 5157 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 5157 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5159 | 41 | 0 | 4.3 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5164 | 20 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5163 | 21 | 0 | 2.2 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✓ | 5163 | 23 | 0 | 2.4 |
