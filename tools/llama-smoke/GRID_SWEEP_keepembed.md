# llama.cpp grid — 2026-08-20 06:39

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

Peak RSS is the Python client only; model/server memory is excluded.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| sweep-keepembed | server | server/server | bars_sc_caltrain_chain | ✗ | 51 | 0 | 0 | 284.4 |
| sweep-keepembed | server | server/server | sky_is_blue_chain | ✗ | 51 | 0 | 0 | 350.9 |
| sweep-keepembed | server | server/server | restaurants_in_sf | ✗ | 51 | 0 | 0 | 117.5 |
| sweep-keepembed | server | server/server | nearby_stories_palo_alto | ✗ | 50 | 0 | 0 | 120.5 |
| sweep-keepembed | server | server/server | tell_me_about_palo_alto | ✗ | 51 | 0 | 0 | 121.5 |
| sweep-keepembed | server | server/server | compare_musk_bezos | ✗ | 51 | 0 | 0 | 117.1 |
| sweep-keepembed | server | server/server | relations_us_iran | ✗ | 51 | 0 | 0 | 117.6 |
| sweep-keepembed | server | server/server | narrate_hp_garage | ✗ | 51 | 0 | 0 | 118.2 |
| sweep-keepembed | server | server/server | what_is_here_in_sf | ✗ | 51 | 0 | 0 | 117.2 |
| sweep-keepembed | server | server/server | putin_biography_chain | ✗ | 51 | 0 | 0 | 344.5 |
| sweep-keepembed | server | server/server | alamo_history_chain | ✗ | 51 | 0 | 0 | 346.7 |
| sweep-keepembed | server | server/server | gravity_waves_creation | ✗ | 51 | 0 | 0 | 120.9 |
| sweep-keepembed | server | server/server | grav_waves_chain | ✗ | 51 | 0 | 0 | 350.2 |
| sweep-keepembed | server | server/server | wwi_vs_wwii_chain | ✗ | 51 | 0 | 0 | 347.8 |
| sweep-keepembed | server | server/server | french_revolution_chain | ✗ | 51 | 0 | 0 | 343.0 |
| sweep-keepembed | server | server/server | crispr_chain | ✗ | 51 | 0 | 0 | 341.6 |
