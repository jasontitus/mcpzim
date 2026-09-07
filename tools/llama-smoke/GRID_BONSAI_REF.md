# llama.cpp grid — 2026-08-20 10:48

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

Peak RSS is the Python client only; model/server memory is excluded.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| bonsai-ref | server | server/server | bars_sc_caltrain_chain | ✓ | 51 | 0 | 0 | 29.3 |
| bonsai-ref | server | server/server | sky_is_blue_chain | ✗ | 50 | 0 | 0 | 16.6 |
| bonsai-ref | server | server/server | restaurants_in_sf | ✓ | 51 | 0 | 0 | 7.8 |
| bonsai-ref | server | server/server | nearby_stories_palo_alto | ✗ | 51 | 0 | 0 | 14.7 |
| bonsai-ref | server | server/server | tell_me_about_palo_alto | ✓ | 50 | 0 | 0 | 4.1 |
| bonsai-ref | server | server/server | compare_musk_bezos | ✓ | 50 | 0 | 0 | 3.7 |
| bonsai-ref | server | server/server | relations_us_iran | ✓ | 50 | 0 | 0 | 4.5 |
| bonsai-ref | server | server/server | narrate_hp_garage | ✓ | 51 | 0 | 0 | 4.1 |
| bonsai-ref | server | server/server | what_is_here_in_sf | ✓ | 51 | 0 | 0 | 3.7 |
| bonsai-ref | server | server/server | putin_biography_chain | ✓ | 51 | 0 | 0 | 12.0 |
| bonsai-ref | server | server/server | alamo_history_chain | ✓ | 50 | 0 | 0 | 10.4 |
| bonsai-ref | server | server/server | gravity_waves_creation | ✓ | 51 | 0 | 0 | 4.7 |
| bonsai-ref | server | server/server | grav_waves_chain | ✗ | 51 | 0 | 0 | 14.1 |
| bonsai-ref | server | server/server | wwi_vs_wwii_chain | ✓ | 51 | 0 | 0 | 17.5 |
| bonsai-ref | server | server/server | french_revolution_chain | ✓ | 51 | 0 | 0 | 14.8 |
| bonsai-ref | server | server/server | crispr_chain | ✗ | 51 | 0 | 0 | 10.9 |
