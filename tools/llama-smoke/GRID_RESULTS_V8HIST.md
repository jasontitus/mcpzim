# llama.cpp grid — 2026-06-10 02:50

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 4136 | 0 | 0 | 3.3 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | sky_is_blue_chain | ✓ | 4164 | 0 | 0 | 3.0 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 4166 | 0 | 0 | 1.8 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 4159 | 0 | 0 | 1.1 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 4161 | 0 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 4158 | 0 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 4158 | 0 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 4159 | 0 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 4158 | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | grav_waves_chain | ✓ | 4164 | 0 | 0 | 2.0 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 4162 | 0 | 0 | 2.3 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | french_revolution_chain | ✓ | 4164 | 0 | 0 | 2.3 |
| lfm2.5-8b-a1b-ft-v8hist-q3km | Q3_K_M | q8_0/q8_0 | crispr_chain | ✓ | 4164 | 0 | 0 | 2.5 |
