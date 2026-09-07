# llama.cpp grid — 2026-08-19 23:48

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

Peak RSS is the Python client only; model/server memory is excluded.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| sweep-all | server | server/server | bars_sc_caltrain_chain | ✗ | 51 | 0 | 0 | 25.1 |
| sweep-all | server | server/server | sky_is_blue_chain | ✗ | 50 | 0 | 0 | 2.7 |
| sweep-all | server | server/server | restaurants_in_sf | ✓ | 50 | 0 | 0 | 24.0 |
| sweep-all | server | server/server | nearby_stories_palo_alto | ✓ | 51 | 0 | 0 | 15.6 |
| sweep-all | server | server/server | tell_me_about_palo_alto | ✓ | 51 | 0 | 0 | 6.6 |
| sweep-all | server | server/server | compare_musk_bezos | ✓ | 51 | 0 | 0 | 7.3 |
| sweep-all | server | server/server | relations_us_iran | ✗ | 51 | 0 | 0 | 4.0 |
| sweep-all | server | server/server | narrate_hp_garage | ✗ | 51 | 0 | 0 | 4.5 |
| sweep-all | server | server/server | what_is_here_in_sf | ✗ | 50 | 0 | 0 | 3.7 |
| sweep-all | server | server/server | putin_biography_chain | ✗ | 50 | 0 | 0 | 15.2 |
| sweep-all | server | server/server | alamo_history_chain | ✗ | 50 | 0 | 0 | 5.8 |
| sweep-all | server | server/server | gravity_waves_creation | ✗ | 50 | 0 | 0 | 4.8 |
| sweep-all | server | server/server | grav_waves_chain | ✓ | 50 | 0 | 0 | 13.4 |
| sweep-all | server | server/server | wwi_vs_wwii_chain | ✗ | 51 | 0 | 0 | 2.6 |
| sweep-all | server | server/server | french_revolution_chain | ✗ | 51 | 0 | 0 | 4.6 |
| sweep-all | server | server/server | crispr_chain | ✗ | 50 | 0 | 0 | 6.2 |
