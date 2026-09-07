# llama.cpp grid — 2026-08-15 11:49

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

Peak RSS is the Python client only; model/server memory is excluded.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| qwen38-gptq-ternary-q2_0 | server | server/server | bars_sc_caltrain_chain | ✗ | 50 | 0 | 0 | 159.7 |
| qwen38-gptq-ternary-q2_0 | server | server/server | sky_is_blue_chain | ✗ | 51 | 0 | 0 | 241.4 |
| qwen38-gptq-ternary-q2_0 | server | server/server | restaurants_in_sf | ✗ | 50 | 0 | 0 | 81.5 |
| qwen38-gptq-ternary-q2_0 | server | server/server | nearby_stories_palo_alto | ✗ | 51 | 0 | 0 | 81.3 |
| qwen38-gptq-ternary-q2_0 | server | server/server | tell_me_about_palo_alto | ✗ | 50 | 0 | 0 | 81.2 |
| qwen38-gptq-ternary-q2_0 | server | server/server | compare_musk_bezos | ✗ | 50 | 0 | 0 | 81.3 |
| qwen38-gptq-ternary-q2_0 | server | server/server | relations_us_iran | ✗ | 51 | 0 | 0 | 81.3 |
| qwen38-gptq-ternary-q2_0 | server | server/server | narrate_hp_garage | ✗ | 50 | 0 | 0 | 81.4 |
| qwen38-gptq-ternary-q2_0 | server | server/server | what_is_here_in_sf | ✗ | 50 | 0 | 0 | 81.5 |
| qwen38-gptq-ternary-q2_0 | server | server/server | putin_biography_chain | ✗ | 51 | 0 | 0 | 241.0 |
| qwen38-gptq-ternary-q2_0 | server | server/server | alamo_history_chain | ✗ | 51 | 0 | 0 | 240.9 |
| qwen38-gptq-ternary-q2_0 | server | server/server | gravity_waves_creation | ✗ | 50 | 0 | 0 | 80.9 |
| qwen38-gptq-ternary-q2_0 | server | server/server | grav_waves_chain | ✗ | 51 | 0 | 0 | 240.9 |
| qwen38-gptq-ternary-q2_0 | server | server/server | wwi_vs_wwii_chain | ✗ | 50 | 0 | 0 | 241.2 |
| qwen38-gptq-ternary-q2_0 | server | server/server | french_revolution_chain | ✗ | 51 | 0 | 0 | 241.3 |
| qwen38-gptq-ternary-q2_0 | server | server/server | crispr_chain | ✗ | 51 | 0 | 0 | 241.1 |
