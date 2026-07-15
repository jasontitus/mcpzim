# llama.cpp grid — 2026-07-14 19:24

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

Peak RSS is the Python client only; model/server memory is excluded.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| bonsai-1bit-27b | server | server/server | putin_biography_chain | ✓ | 51 | 0 | 0 | 18.0 |
| bonsai-1bit-27b | server | server/server | alamo_history_chain | ✓ | 51 | 0 | 0 | 11.5 |
| bonsai-1bit-27b | server | server/server | gravity_waves_creation | ✗ | 51 | 0 | 0 | 3.0 |
