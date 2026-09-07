# llama.cpp grid — 2026-08-18 02:31

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

Peak RSS is the Python client only; model/server memory is excluded.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| a6-eval | server | server/server | bars_sc_caltrain_chain | ✗ | 51 | 0 | 0 | 30.8 |
| a6-eval | server | server/server | sky_is_blue_chain | ✗ | 51 | 0 | 0 | 9.8 |
| a6-eval | server | server/server | restaurants_in_sf | ✓ | 51 | 0 | 0 | 4.8 |
| a6-eval | server | server/server | nearby_stories_palo_alto | ✗ | 51 | 0 | 0 | 2.9 |
| a6-eval | server | server/server | tell_me_about_palo_alto | ✗ | 51 | 0 | 0 | 3.6 |
| a6-eval | server | server/server | compare_musk_bezos | ✗ | 51 | 0 | 0 | 50.0 |
| a6-eval | server | server/server | relations_us_iran | ✗ | 51 | 0 | 0 | 4.7 |
| a6-eval | server | server/server | narrate_hp_garage | ✗ | 51 | 0 | 0 | 8.0 |
| a6-eval | server | server/server | what_is_here_in_sf | ✗ | 51 | 0 | 0 | 24.2 |
| a6-eval | server | server/server | putin_biography_chain | ✗ | 51 | 0 | 0 | 23.6 |
| a6-eval | server | server/server | alamo_history_chain | ✗ | 51 | 0 | 0 | 34.4 |
| a6-eval | server | server/server | gravity_waves_creation | ✗ | 51 | 0 | 0 | 60.1 |
| a6-eval | server | server/server | grav_waves_chain | ✗ | 50 | 0 | 0 | 46.0 |
| a6-eval | server | server/server | wwi_vs_wwii_chain | ✗ | 51 | 0 | 0 | 36.7 |
| a6-eval | server | server/server | french_revolution_chain | ✗ | 51 | 0 | 0 | 8.2 |
| a6-eval | server | server/server | crispr_chain | ✗ | 51 | 0 | 0 | 60.6 |
