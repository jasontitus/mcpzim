# llama.cpp grid — 2026-07-14 19:08

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

Peak RSS is the Python client only; model/server memory is excluded.

Palo Alto, Putin, and Alamo were rerun with corrected stub evidence
contracts at 19:18 using the same model, seed, and runtime configuration.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| bonsai-ternary-27b | server | server/server | bars_sc_caltrain_chain | ✓ | 51 | 0 | 0 | 48.0 |
| bonsai-ternary-27b | server | server/server | sky_is_blue_chain | ✓ | 51 | 0 | 0 | 41.0 |
| bonsai-ternary-27b | server | server/server | restaurants_in_sf | ✓ | 51 | 0 | 0 | 9.7 |
| bonsai-ternary-27b | server | server/server | nearby_stories_palo_alto | ✓ | 51 | 0 | 0 | 13.1 |
| bonsai-ternary-27b | server | server/server | tell_me_about_palo_alto | ✓ | 51 | 0 | 0 | 9.1 |
| bonsai-ternary-27b | server | server/server | compare_musk_bezos | ✓ | 51 | 0 | 0 | 13.5 |
| bonsai-ternary-27b | server | server/server | relations_us_iran | ✓ | 51 | 0 | 0 | 3.6 |
| bonsai-ternary-27b | server | server/server | narrate_hp_garage | ✓ | 51 | 0 | 0 | 4.5 |
| bonsai-ternary-27b | server | server/server | what_is_here_in_sf | ✓ | 51 | 0 | 0 | 8.7 |
| bonsai-ternary-27b | server | server/server | putin_biography_chain | ✓ | 51 | 0 | 0 | 20.6 |
| bonsai-ternary-27b | server | server/server | alamo_history_chain | ✓ | 51 | 0 | 0 | 20.1 |
| bonsai-ternary-27b | server | server/server | gravity_waves_creation | ✓ | 51 | 0 | 0 | 13.2 |
| bonsai-ternary-27b | server | server/server | grav_waves_chain | ✓ | 51 | 0 | 0 | 33.3 |
| bonsai-ternary-27b | server | server/server | wwi_vs_wwii_chain | ✓ | 51 | 0 | 0 | 63.3 |
| bonsai-ternary-27b | server | server/server | french_revolution_chain | ✓ | 51 | 0 | 0 | 46.7 |
| bonsai-ternary-27b | server | server/server | crispr_chain | ✓ | 52 | 0 | 0 | 28.3 |
