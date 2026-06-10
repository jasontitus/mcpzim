# llama.cpp grid — gemma4-e4b-qat (official Google QAT Q4_0) — 2026-06-09

Full 13-scenario run, KV q8_0/q8_0 (same config as the LFM2.5-FT 12/13 run).
compare_musk_bezos / relations_us_iran / wwi_vs_wwii_chain initially CRASHED the
harness (model emitted compare_articles arguments as a bare JSON LIST);
dispatch_tool now normalizes that shape and those rows are the scored re-runs.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5313 | 101 | 0 | 10.9 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5320 | 74 | 0 | 7.8 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | restaurants_in_sf | ✓ | 5304 | 19 | 0 | 2.0 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5304 | 34 | 0 | 3.5 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5306 | 19 | 0 | 1.9 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | compare_musk_bezos | ✓ | 5303 | 17 | 0 | 1.7 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | relations_us_iran | ✓ | 5303 | 20 | 0 | 2.0 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | narrate_hp_garage | ✓ | 5302 | 20 | 0 | 2.0 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | what_is_here_in_sf | ✓ | 5303 | 18 | 0 | 1.9 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | grav_waves_chain | ✗ | 5315 | 48 | 0 | 5.1 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5314 | 51 | 0 | 5.4 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | french_revolution_chain | ✗ | 5317 | 45 | 0 | 4.7 |
| gemma4-e4b-qat | Q4_0 | q8_0/q8_0 | crispr_chain | ✗ | 5316 | 59 | 0 | 6.3 |

Score: 8/13 pass. All 5 fails are the multi-turn knowledge chains.
