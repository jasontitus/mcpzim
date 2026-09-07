# llama.cpp grid — 2026-08-20 00:01

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

Peak RSS is the Python client only; model/server memory is excluded.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| sweep-keep00_15 | server | server/server | bars_sc_caltrain_chain | ✗ | 51 | 0 | 0 | 290.7 |
| sweep-keep00_15 | server | server/server | sky_is_blue_chain | ✗ | 51 | 0 | 0 | 355.8 |
| sweep-keep00_15 | server | server/server | restaurants_in_sf | ✗ | 51 | 0 | 0 | 115.0 |
| sweep-keep00_15 | server | server/server | nearby_stories_palo_alto | ✗ | 50 | 0 | 0 | 45.3 |
| sweep-keep00_15 | server | server/server | tell_me_about_palo_alto | ✗ | 51 | 0 | 0 | 124.7 |
| sweep-keep00_15 | server | server/server | compare_musk_bezos | ✗ | 51 | 0 | 0 | 119.3 |
| sweep-keep00_15 | server | server/server | relations_us_iran | ✗ | 50 | 0 | 0 | 124.3 |
| sweep-keep00_15 | server | server/server | narrate_hp_garage | ✗ | 51 | 0 | 0 | 118.9 |
| sweep-keep00_15 | server | server/server | what_is_here_in_sf | ✗ | 51 | 0 | 0 | 120.4 |
| sweep-keep00_15 | server | server/server | putin_biography_chain | ✗ | 52 | 0 | 0 | 246.6 |
| sweep-keep00_15 | server | server/server | alamo_history_chain | ✗ | 51 | 0 | 0 | 302.4 |
| sweep-keep00_15 | server | server/server | gravity_waves_creation | ✗ | 51 | 0 | 0 | 126.7 |
| sweep-keep00_15 | server | server/server | grav_waves_chain | ✗ | 51 | 0 | 0 | 147.7 |
| sweep-keep00_15 | server | server/server | wwi_vs_wwii_chain | ✗ | 51 | 0 | 0 | 244.1 |
| sweep-keep00_15 | server | server/server | french_revolution_chain | ✗ | 51 | 0 | 0 | 375.9 |
| sweep-keep00_15 | server | server/server | crispr_chain | ✗ | 51 | 0 | 0 | 363.3 |
