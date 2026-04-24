# Sequential Gemma 3 4B quant sweep — 2026-04-23

```
llama_context: n_ctx_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
eval: scenario=bars_sc_caltrain_chain
       model=bartowski/google_gemma-3-4b-it-GGUF/google_gemma-3-4b-it-Q4_K_M.gguf
       cache_k=q8_0 cache_v=q8_0 flash_attn=True
       load: 1.42s · rss: 3515 MB

RESULT scenario=bars_sc_caltrain_chain passed=True wall_s=12.3
RESULT peak_mb=3567 ge5gb=0 ge6gb=0 ge7gb=0 samples=130
  {'turn': 0, 'iter': 0, 't_s': 1.72, 'tool_calls': ['near_named_place'], 'content_preview': '```tool_call\n{"function":"near_named_place","parameters":{"place":"San Carlos","radius_km":5,"kinds":["bar","restaurant"]}}\n```'}
  {'turn': 0, 'iter': 1, 't_s': 5.51, 'tool_calls': [], 'content_preview': 'There are 25 bars in San Carlos, ranging from 200 to 3800 meters away.'}
  {'turn': 0, 'final_content': 'There are 25 bars in San Carlos, ranging from 200 to 3800 meters away.', 'tool_calls_seen': ['near_named_place'], 'tools_ok': True, 'response_ok': True}
  {'turn': 1, 'iter': 0, 't_s': 2.12, 'tool_calls': ['near_places'], 'content_preview': '```tool_call\n{"function":"near_places","parameters":{"lat":37.5124,"lon":-122.2606,"place":"Caltrain","radius_km":1,"kinds":["bar"]}}\n```'}
  {'turn': 1, 'iter': 1, 't_s': 2.95, 'tool_calls': [], 'content_preview': 'The closest bar to Caltrain is Little Sky Bakery, located about 129 meters away.'}
  {'turn': 1, 'final_content': 'The closest bar to Caltrain is Little Sky Bakery, located about 129 meters away.', 'tool_calls_seen': ['near_places'], 'tools_ok': True, 'response_ok': True}

llama_context: n_ctx_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
eval: scenario=bars_sc_caltrain_chain
       model=bartowski/google_gemma-3-4b-it-GGUF/google_gemma-3-4b-it-Q5_K_M.gguf
       cache_k=q8_0 cache_v=q8_0 flash_attn=True
       load: 1.53s · rss: 3832 MB

RESULT scenario=bars_sc_caltrain_chain passed=True wall_s=13.7
RESULT peak_mb=3883 ge5gb=0 ge6gb=0 ge7gb=0 samples=146
  {'turn': 0, 'iter': 0, 't_s': 2.09, 'tool_calls': ['near_named_place'], 'content_preview': '```tool_call\n{"function":"near_named_place","parameters":{"place":"San Carlos","radius_km":5,"kinds":["bar","pub","restaurant"]}}\n```'}
  {'turn': 0, 'iter': 1, 't_s': 5.67, 'tool_calls': [], 'content_preview': 'There are 25 bars in San Carlos, California.'}
  {'turn': 0, 'final_content': 'There are 25 bars in San Carlos, California.', 'tool_calls_seen': ['near_named_place'], 'tools_ok': True, 'response_ok': True}
  {'turn': 1, 'iter': 0, 't_s': 2.57, 'tool_calls': ['near_places'], 'content_preview': '```tool_call\n{"function":"near_places","parameters":{"lat":37.5124,"lon":-122.2606,"place":"Caltrain","radius_km":1,"kinds":["bar"]}}\n```'}
  {'turn': 1, 'iter': 1, 't_s': 3.36, 'tool_calls': [], 'content_preview': 'Little Sky Bakery is closest to Caltrain, about 129 meters away.'}
  {'turn': 1, 'final_content': 'Little Sky Bakery is closest to Caltrain, about 129 meters away.', 'tool_calls_seen': ['near_places'], 'tools_ok': True, 'response_ok': True}

llama_context: n_ctx_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
eval: scenario=bars_sc_caltrain_chain
       model=bartowski/google_gemma-3-4b-it-GGUF/google_gemma-3-4b-it-Q6_K.gguf
       cache_k=q8_0 cache_v=q8_0 flash_attn=True
       load: 1.81s · rss: 3958 MB

RESULT scenario=bars_sc_caltrain_chain passed=True wall_s=11.9
RESULT peak_mb=4013 ge5gb=0 ge6gb=0 ge7gb=0 samples=130
  {'turn': 0, 'iter': 0, 't_s': 1.43, 'tool_calls': ['near_named_place'], 'content_preview': '```tool_call\n{"function":"near_named_place","parameters":{"place":"San Carlos","radius_km":5,"kinds":["bar"]}}\n```'}
  {'turn': 0, 'iter': 1, 't_s': 5.23, 'tool_calls': [], 'content_preview': 'There are 25 bars near San Carlos, California.'}
  {'turn': 0, 'final_content': 'There are 25 bars near San Carlos, California.', 'tool_calls_seen': ['near_named_place'], 'tools_ok': True, 'response_ok': True}
  {'turn': 1, 'iter': 0, 't_s': 2.19, 'tool_calls': ['near_places'], 'content_preview': '```tool_call\n{"function":"near_places","parameters":{"lat":37.5124,"lon":-122.2606,"place":"Caltrain","radius_km":1,"kinds":["bar"]}}\n```'}
  {'turn': 1, 'iter': 1, 't_s': 3.01, 'tool_calls': [], 'content_preview': 'Refuge Belgian is closest to Caltrain, about 324 meters away.'}
  {'turn': 1, 'final_content': 'Refuge Belgian is closest to Caltrain, about 324 meters away.', 'tool_calls_seen': ['near_places'], 'tools_ok': True, 'response_ok': True}

llama_context: n_ctx_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
eval: scenario=bars_sc_caltrain_chain
       model=bartowski/google_gemma-3-4b-it-GGUF/google_gemma-3-4b-it-Q8_0.gguf
       cache_k=q8_0 cache_v=q8_0 flash_attn=True
       load: 2.33s · rss: 4633 MB

RESULT scenario=bars_sc_caltrain_chain passed=True wall_s=12.0
RESULT peak_mb=4689 ge5gb=0 ge6gb=0 ge7gb=0 samples=137
  {'turn': 0, 'iter': 0, 't_s': 1.46, 'tool_calls': ['near_named_place'], 'content_preview': '```tool_call\n{"function":"near_named_place","parameters":{"place":"San Carlos","radius_km":5,"kinds":["bar","restaurant"]}}\n```'}
  {'turn': 0, 'iter': 1, 't_s': 5.01, 'tool_calls': [], 'content_preview': 'There are 25 bars in San Carlos, within a 5km radius.'}
  {'turn': 0, 'final_content': 'There are 25 bars in San Carlos, within a 5km radius.', 'tool_calls_seen': ['near_named_place'], 'tools_ok': True, 'response_ok': True}
  {'turn': 1, 'iter': 0, 't_s': 2.7, 'tool_calls': ['near_places'], 'content_preview': '```tool_call\n{"function":"near_places","parameters":{"lat":37.5124,"lon":-122.2606,"place":"Caltrain","radius_km":1,"kinds":["bar"]}}\n```'}
  {'turn': 1, 'iter': 1, 't_s': 2.8, 'tool_calls': [], 'content_preview': 'Little Sky Bakery is closest to Caltrain, about 129 meters away.'}
  {'turn': 1, 'final_content': 'Little Sky Bakery is closest to Caltrain, about 129 meters away.', 'tool_calls_seen': ['near_places'], 'tools_ok': True, 'response_ok': True}

```
