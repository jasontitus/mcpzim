# llama.cpp grid — 2026-05-29 00:05

Running sequentially — each combo is its own python subprocess so peak-RSS numbers don't carry over.

| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |
|---|---|---|---|---|---|---|---|---|
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5171 | 25 | 0 | 2.6 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✓ | 5162 | 20 | 0 | 2.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✓ | 5164 | 17 | 0 | 1.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5157 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5156 | 10 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5156 | 10 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 5156 | 9 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 5158 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 5156 | 8 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5162 | 20 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✓ | 5162 | 15 | 0 | 1.5 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5175 | 36 | 0 | 3.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q8_0/q8_0 | crispr_chain | ✓ | 5164 | 20 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | bars_sc_caltrain_chain | ✓ | 5151 | 35 | 0 | 3.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | sky_is_blue_chain | ✗ | 5140 | 22 | 0 | 2.3 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | restaurants_in_sf | ✓ | 5140 | 16 | 0 | 1.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | nearby_stories_palo_alto | ✗ | 5136 | 17 | 0 | 1.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | tell_me_about_palo_alto | ✓ | 5133 | 9 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | compare_musk_bezos | ✓ | 5133 | 10 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | relations_us_iran | ✓ | 5132 | 8 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | narrate_hp_garage | ✗ | 5132 | 8 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | what_is_here_in_sf | ✓ | 5132 | 7 | 0 | 0.7 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | grav_waves_chain | ✓ | 5140 | 20 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | wwi_vs_wwii_chain | ✓ | 5138 | 20 | 0 | 2.0 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | french_revolution_chain | ✓ | 5138 | 24 | 0 | 2.5 |
| lfm2.5-8b-a1b-ft | Q4_K_M | q4_0/q4_0 | crispr_chain | ✗ | 5139 | 21 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 5170 | 44 | 0 | 4.7 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 5179 | 184 | 0 | 19.9 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 5162 | 21 | 0 | 2.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 5158 | 24 | 0 | 2.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 5159 | 14 | 0 | 1.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 5159 | 23 | 0 | 2.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | relations_us_iran | ✗ | 5164 | 61 | 0 | 6.6 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | narrate_hp_garage | ✓ | 5158 | 14 | 0 | 1.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | what_is_here_in_sf | ✗ | 5158 | 11 | 0 | 1.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 5178 | 216 | 0 | 23.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 5179 | 178 | 0 | 19.3 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 5177 | 182 | 0 | 19.8 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q8_0/q8_0 | crispr_chain | ✗ | 5169 | 88 | 0 | 9.5 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | bars_sc_caltrain_chain | ✓ | 5151 | 75 | 0 | 8.0 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | sky_is_blue_chain | ✓ | 5150 | 115 | 0 | 12.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | restaurants_in_sf | ✗ | 5138 | 25 | 0 | 2.6 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | nearby_stories_palo_alto | ✗ | 5138 | 58 | 0 | 6.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | tell_me_about_palo_alto | ✓ | 5134 | 13 | 0 | 1.3 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | compare_musk_bezos | ✓ | 5134 | 23 | 0 | 2.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | relations_us_iran | ✗ | 5139 | 61 | 0 | 6.6 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | narrate_hp_garage | ✓ | 5134 | 23 | 0 | 2.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | what_is_here_in_sf | ✗ | 5134 | 11 | 0 | 1.1 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | grav_waves_chain | ✗ | 5142 | 38 | 0 | 4.0 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | wwi_vs_wwii_chain | ✗ | 5155 | 179 | 0 | 19.4 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | french_revolution_chain | ✗ | 5157 | 205 | 0 | 22.3 |
| lfm2.5-8b-a1b-ft-native | Q4_K_M | q4_0/q4_0 | crispr_chain | ✗ | 5152 | 142 | 0 | 15.5 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | sky_is_blue_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | restaurants_in_sf | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | nearby_stories_palo_alto | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | tell_me_about_palo_alto | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | compare_musk_bezos | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | relations_us_iran | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | narrate_hp_garage | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | what_is_here_in_sf | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | grav_waves_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | french_revolution_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q8_0/q8_0 | crispr_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | bars_sc_caltrain_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | sky_is_blue_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | restaurants_in_sf | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | nearby_stories_palo_alto | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | tell_me_about_palo_alto | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | compare_musk_bezos | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | relations_us_iran | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | narrate_hp_garage | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | what_is_here_in_sf | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | grav_waves_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | wwi_vs_wwii_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | french_revolution_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q3km | Q3_K_M | q4_0/q4_0 | crispr_chain | · | rc=1:     self.close() /   File "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 79, in close /     if self.sampler is not None: /        ^^^^^^^^^^^^ / AttributeError: 'LlamaModel' object has no attribute 'sampler' | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 3269 | 0 | 0 | 2.2 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | sky_is_blue_chain | ✗ | 3291 | 0 | 0 | 2.8 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | restaurants_in_sf | ✓ | 3282 | 0 | 0 | 0.9 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | nearby_stories_palo_alto | ✗ | 3281 | 0 | 0 | 0.6 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 3282 | 0 | 0 | 0.6 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | compare_musk_bezos | ✗ | 3281 | 0 | 0 | 0.5 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | relations_us_iran | ✗ | 3282 | 0 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | narrate_hp_garage | ✗ | 3281 | 0 | 0 | 0.8 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | what_is_here_in_sf | ✗ | 3281 | 0 | 0 | 0.7 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | grav_waves_chain | ✗ | 3291 | 0 | 0 | 3.5 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 3284 | 0 | 0 | 1.2 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | french_revolution_chain | ✗ | 3286 | 0 | 0 | 1.8 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q8_0/q8_0 | crispr_chain | ✗ | 3286 | 0 | 0 | 2.0 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | bars_sc_caltrain_chain | ✗ | 3261 | 0 | 0 | 1.3 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | sky_is_blue_chain | ✗ | 3262 | 0 | 0 | 1.6 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | restaurants_in_sf | ✗ | 3257 | 0 | 0 | 0.6 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | nearby_stories_palo_alto | ✗ | 3257 | 0 | 0 | 0.6 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | tell_me_about_palo_alto | ✗ | 3257 | 0 | 0 | 1.4 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | compare_musk_bezos | ✗ | 3257 | 0 | 0 | 0.5 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | relations_us_iran | ✗ | 3258 | 0 | 0 | 16.6 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | narrate_hp_garage | ✗ | 3258 | 0 | 0 | 1.0 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | what_is_here_in_sf | ✗ | 3257 | 0 | 0 | 0.6 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | grav_waves_chain | ✗ | 3262 | 0 | 0 | 1.6 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | wwi_vs_wwii_chain | ✗ | 3262 | 0 | 0 | 2.3 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | french_revolution_chain | ✗ | 3264 | 0 | 0 | 1.6 |
| lfm2.5-8b-a1b-ft-q2k | Q2_K | q4_0/q4_0 | crispr_chain | ✗ | 3266 | 0 | 0 | 18.5 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 3838 | 0 | 0 | 3.0 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | sky_is_blue_chain | ✗ | 4164 | 0 | 0 | 2.5 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | restaurants_in_sf | ✗ | 4165 | 0 | 0 | 1.9 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 4158 | 0 | 0 | 1.2 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | tell_me_about_palo_alto | ✓ | 4158 | 0 | 0 | 0.8 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | compare_musk_bezos | ✓ | 4158 | 0 | 0 | 0.9 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | relations_us_iran | ✓ | 4158 | 0 | 0 | 1.0 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | narrate_hp_garage | ✗ | 4158 | 0 | 0 | 0.8 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | what_is_here_in_sf | ✓ | 4159 | 0 | 0 | 0.8 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | grav_waves_chain | ✗ | 4161 | 0 | 0 | 2.6 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | wwi_vs_wwii_chain | ✓ | 4163 | 0 | 0 | 2.0 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | french_revolution_chain | ✗ | 4164 | 0 | 0 | 1.9 |
| lfm2.5-v6-q3km | Q3_K_M | q8_0/q8_0 | crispr_chain | ✓ | 4164 | 0 | 0 | 2.2 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | bars_sc_caltrain_chain | ✓ | 4148 | 0 | 0 | 3.0 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | sky_is_blue_chain | ✗ | 4138 | 0 | 0 | 1.7 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | restaurants_in_sf | ✗ | 4142 | 0 | 0 | 1.8 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | nearby_stories_palo_alto | ✓ | 4135 | 0 | 0 | 1.3 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | tell_me_about_palo_alto | ✓ | 4134 | 0 | 0 | 1.0 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | compare_musk_bezos | ✗ | 4132 | 0 | 0 | 0.5 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | relations_us_iran | ✗ | 4133 | 0 | 0 | 0.5 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | narrate_hp_garage | ✗ | 4134 | 0 | 0 | 0.9 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | what_is_here_in_sf | ✓ | 4134 | 0 | 0 | 0.8 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | grav_waves_chain | ✗ | 4136 | 0 | 0 | 2.2 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | wwi_vs_wwii_chain | ✓ | 4140 | 0 | 0 | 2.3 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | french_revolution_chain | ✓ | 4140 | 0 | 0 | 1.9 |
| lfm2.5-v6-q3km | Q3_K_M | q4_0/q4_0 | crispr_chain | ✗ | 4140 | 0 | 0 | 2.0 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | bars_sc_caltrain_chain | ✓ | 3250 | 0 | 0 | 2.3 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | sky_is_blue_chain | ✗ | 3285 | 0 | 0 | 3.2 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | restaurants_in_sf | ✓ | 3282 | 0 | 0 | 0.9 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | nearby_stories_palo_alto | ✓ | 3283 | 0 | 0 | 1.3 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | tell_me_about_palo_alto | ✗ | 3281 | 0 | 0 | 0.5 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | compare_musk_bezos | ✗ | 3281 | 0 | 0 | 1.1 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | relations_us_iran | ✗ | 3281 | 0 | 0 | 0.7 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | narrate_hp_garage | ✗ | 3282 | 0 | 0 | 1.8 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | what_is_here_in_sf | ✗ | 3281 | 0 | 0 | 0.4 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | grav_waves_chain | ✗ | 3296 | 0 | 0 | 5.5 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | wwi_vs_wwii_chain | ✗ | 3289 | 0 | 0 | 4.7 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | french_revolution_chain | ✗ | 3284 | 0 | 0 | 1.8 |
| lfm2.5-v6-q2k | Q2_K | q8_0/q8_0 | crispr_chain | ✗ | 3284 | 0 | 0 | 1.7 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | bars_sc_caltrain_chain | ✓ | 3273 | 0 | 0 | 5.2 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | sky_is_blue_chain | ✗ | 3262 | 0 | 0 | 1.5 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | restaurants_in_sf | ✓ | 3258 | 0 | 0 | 0.9 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | nearby_stories_palo_alto | ✗ | 3257 | 0 | 0 | 0.5 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | tell_me_about_palo_alto | ✗ | 3257 | 0 | 0 | 0.6 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | compare_musk_bezos | ✗ | 3257 | 0 | 0 | 0.9 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | relations_us_iran | ✗ | 3257 | 0 | 0 | 0.8 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | narrate_hp_garage | ✗ | 3258 | 0 | 0 | 16.9 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | what_is_here_in_sf | ✓ | 3259 | 0 | 0 | 16.9 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | grav_waves_chain | ✗ | 3264 | 0 | 0 | 2.8 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | wwi_vs_wwii_chain | ✗ | 3261 | 0 | 0 | 4.3 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | french_revolution_chain | ✗ | 3265 | 0 | 0 | 3.7 |
| lfm2.5-v6-q2k | Q2_K | q4_0/q4_0 | crispr_chain | ✗ | 3261 | 0 | 0 | 2.7 |
