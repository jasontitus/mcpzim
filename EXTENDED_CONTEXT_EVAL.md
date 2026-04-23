# Extended-context conversational evals — design

The app today scores well on one-shot "pick the right tool" tests
(`ios/MCPZimEval/EvalHarness.swift`), but our *actual* target
interaction model is the walking-with-headphones case:

> I'm on a walk, phone in my pocket, AirPods in. I ask about a topic
> I don't know well — gravitational waves, French Revolution, why the
> sky is blue — and the app fetches Wikipedia articles, reads the
> relevant sections, and answers. Then I ask a follow-up. Then
> another. Over 5-15 minutes the conversation drifts across related
> topics.

Whether that's plausible on a phone is mostly a context + memory
question: each article section is ~1-3 k tokens, the app's system
preamble is ~7 k, and 5-10 follow-ups routinely push the conversation
past 20-30 k tokens. This doc lays out an eval suite that exercises
the long-context conversational path end-to-end so we can measure the
ceiling on each device.

## 1. Scenario taxonomy

Ten concrete topics the app should be able to sustain a 4-6 turn
conversation about, grouped by character:

### Physics / science

- **"Why is the sky blue?"** — Rayleigh scattering → extends to
  sunset red, ocean blue, Tyndall effect.
- **"Recent advances in gravitational wave research"** — LIGO, Virgo,
  detector interferometry, multi-messenger astronomy, 2017 neutron-
  star collision, squeezed-light improvements.
- **"How do lithium-ion batteries work?"** — cathode / anode
  chemistry, solid electrolyte interphase, thermal runaway, next-gen
  solid-state candidates, supply-chain dependencies.
- **"What is CRISPR and how does it edit DNA?"** — Cas9 mechanism,
  guide RNA, off-target effects, therapeutic applications (sickle
  cell), germline ethics.

### History / global relations

- **"Compare WWI and WWII — causes, scale, how each ended."**
  Requires two long articles, cross-references. Casualty figures are
  a factual anchor.
- **"How did the French Revolution unfold?"** — Causes,
  Estates-General, Bastille, Reign of Terror, Directory,
  Napoleon. Natural 4-turn expansion.
- **"US-China relations since 1972"** — Nixon opening, trade
  normalization, WTO accession, Xi era, Taiwan tension. Extends to
  semiconductor export controls, climate cooperation.

### Ideas / cultural

- **"Explain general relativity in plain terms, then differential
  geometry, then gravitational lensing."** Tests the model's ability
  to layer abstraction across turns.
- **"Walk me through the history of the internet — ARPANET, TCP/IP,
  the web, the mobile web, today."** Natural chronological chain.
- **"Why did the Roman Empire fall?"** — competing theses (Gibbon vs
  Heather vs Ward-Perkins), ask for each, then a synthesis.

## 2. Turn shape

Each scenario is a sequence of 4–6 user turns. The expected
tool-call patterns fall into four categories:

| Turn kind | Example | Expected tools | Context growth |
|---|---|---|---|
| `opener` | "Why is the sky blue?" | `article_overview("Rayleigh scattering")` or `search` → `get_article_section("lead")` | +1-3 k |
| `expand` | "Tell me more about sunsets." | `get_article_section("Color of sunsets")` — ideally reusing the already-retrieved article | +1-2 k |
| `crossref` | "How does that compare to Tyndall scattering?" | `compare_articles("Rayleigh", "Tyndall")` or two `get_article_section` calls | +2-4 k |
| `clarify` | "Wait, so wavelength controls it?" | *No tool call.* Answer from the cached sections. | +0 |

Good scenarios mix these — a `clarify` turn after an `expand` is the
sharpest test of whether the model is carrying prior context or
re-retrieving.

## 3. Scoring

Extend the existing `TurnExpect` with two new predicates:

```swift
struct TurnExpect {
    var toolsCalledAny: [String]
    var toolsNotCalled: [String]
    var responseIncludesAny: [String]
    var responseExcludes: [String]

    // NEW: on `clarify` turns we want to assert NO tool call happened
    // (the model should have answered from cached context).
    var requireNoToolCall: Bool = false

    // NEW: the reply must reference a section title or content snippet
    // from a previous turn's `get_article_section` result — confirms
    // context was carried forward rather than recomputed.
    var referencesPriorSection: String? = nil
}
```

Plus a per-scenario ceiling:

```swift
struct Scenario {
    // NEW: peak-RSS cap; scenario fails if MLX peak exceeds this.
    // Phone-target scenarios set ~5.5 GB; mac-only scenarios set ~12 GB.
    var maxPeakMB: Int? = nil
}
```

## 4. Device budget targets

From `ON_DEVICE_MODEL_REPORT_2026-04-23.md` §4:

| Device | Safe model-peak | Max prompt tokens |
|---|---|---|
| iPhone 17 Pro Max (6 GB jetsam) | ≤5.5 GB | ~40 k (Gemma 3) / ~20 k (Qwen 3) |
| iPad Pro (M-series, 16 GB) | ≤10 GB | ~80 k (Gemma 3) |
| Mac (64 GB+) | ≤20 GB | ~200 k for 4 B models |

A "walking with headphones" conversation probably averages 4-6 turns,
~2 k of article text per turn, plus the 7 k preamble → 15-20 k
tokens. That fits Gemma 3 at 4.4 GB peak comfortably on phone. Where
it breaks:

- Five sections fetched per turn × 6 turns = 30-40 k. Borderline.
- `compare_articles` pulls two full overviews — each turn is
  ~5-7 k on its own. Three of those in a row = 20-30 k.
- The `clarify → expand → compare` pattern can snowball to 50 k if
  the model re-retrieves instead of reusing cached sections.

So the eval needs to *specifically* exercise the snowball case and
report the peak token count per turn as a second-class metric.

## 5. Phone viability thesis

Based on the memory curves in the bench report:

- **Today**: Gemma 3 4B IT (4-bit, text) handles 40 k-token
  conversations on an iPhone 17 Pro Max (5.41 GB peak). That covers
  ~20 total article sections across a conversation — plenty for a
  15-minute topical walk. This is the one on-device candidate that
  actually fits.
- **In 12 months**: iPhone jetsam likely bumps from 6 GB to 8 GB
  (historical pattern). Combined with the 1.5x MLX improvements on
  M5/M6 we'd bank on current Mac-class contexts (100 k+) being
  phone-viable. So validating the *workflow* on Mac today is how we
  future-proof it for phone tomorrow.
- **Rules that won't change**: Qwen 3 + kv4 scales disastrously at
  long contexts (Δ per 10 k tokens = 2.9 GB, vs Gemma 3's 500 MB).
  Any long-context eval we build will keep Qwen+kv4 below 20 k
  forever. Gemma 3's sliding-window attention is the right
  architecture for this use case.

## 6. Implementation plan

Phase A — *today* (Mac-first):

1. Add the 10 scenarios as `Scenario` entries in
   `ios/MCPZimEval/EvalHarness.swift`. Each scenario has 4-6
   turns with `TurnExpect` extended as in §3.
2. Add `requireNoToolCall` + `referencesPriorSection` to
   `TurnExpect` and implement the checks in the harness's per-turn
   scoring loop.
3. Run `MCPZimEvalCLI --variant gemma3-4b-it-text-4bit` across the
   new scenarios on Mac. Report pass/fail + peak tokens/peak MB.
4. Compare against `mlx-community/gemma-3-text-12b-it-4bit` on Mac
   only as the ceiling.

Phase B — phone:

5. Run the same scenarios on-device via the iOS build + debug
   report. Log per-turn token count + MLX peak.
6. Tune per-turn section-size / article-caps
   (`DeviceProfile.articleCapKB`) to stay under the 5.5 GB ceiling
   without losing grounding.

Phase C — snowball + clarify stress:

7. Build adversarial scenarios designed to trip the snowball:
   "Tell me about X." "And Y." "How does Y compare to Z?" "What
   did you say about X earlier?" The last turn tests clarify
   (no tool call) + section reuse.
8. Log whether the model re-retrieves or reuses; ideally we want
   context-reuse rates above 70%.

## 7. Starter scenarios (skeletons)

Quick sketch of what a few look like in harness form — stub only,
`article_overview` / `get_article_section` calls intentionally
abstract over which ZIM serves them:

```swift
.init(name: "sky_is_blue_chain",
      turns: [
        (user: "Why is the sky blue?",
         expect: .init(
            toolsCalledAny: ["article_overview"],
            responseIncludesAny: ["rayleigh", "scatter", "wavelength"]
         )),
        (user: "So why are sunsets red then?",
         expect: .init(
            toolsCalledAny: ["get_article_section", "article_overview"],
            responseIncludesAny: ["longer", "wavelength", "atmosphere"]
         )),
        (user: "Wait, what controls which wavelength wins?",
         expect: .init(
            toolsCalledAny: [],
            requireNoToolCall: true,
            responseIncludesAny: ["wavelength", "path", "angle"]
         )),
      ]),

.init(name: "gravitational_waves_recent",
      turns: [
        (user: "What are gravitational waves?", expect: .init(
            toolsCalledAny: ["article_overview"],
            responseIncludesAny: ["spacetime", "einstein", "ligo"])),
        (user: "Tell me about LIGO's detectors.", expect: .init(
            toolsCalledAny: ["article_overview", "get_article_section"],
            responseIncludesAny: ["interferometer", "hanford", "livingston"])),
        (user: "How was the 2017 neutron-star collision different?",
         expect: .init(
            toolsCalledAny: ["search", "get_article_section"],
            responseIncludesAny: ["gw170817", "kilonova", "electromagnetic"])),
        (user: "Multi-messenger astronomy — what's the big idea?",
         expect: .init(
            toolsCalledAny: [],
            requireNoToolCall: true,
            responseIncludesAny: ["photons", "gravitational", "neutrinos"])),
      ]),

.init(name: "wwi_vs_wwii",
      turns: [
        (user: "Compare World War I and World War II — causes and scale.",
         expect: .init(
            toolsCalledAny: ["compare_articles"],
            responseIncludesAny: ["1914", "1939", "alliance", "fascism"])),
        (user: "How many people were killed in each?",
         expect: .init(
            toolsCalledAny: ["get_article_section"],
            responseIncludesAny: ["million", "civilian", "casualt"])),
        (user: "What changed between the two that made WWII so much more deadly?",
         expect: .init(
            toolsCalledAny: [],
            requireNoToolCall: true,
            responseIncludesAny: ["industrial", "air", "civilian"])),
      ]),
```

## 8. What this doesn't try to do

- **Real-time voice latency** — covered separately by the Kokoro +
  VoiceChat work. This eval only cares about text latency.
- **Creative / stylistic quality** — these scenarios are factual.
  Measuring prose quality requires a human-judged or LM-judge
  harness which isn't plumbed yet.
- **Multilingual** — all scenarios are English. The ZIM corpus is
  English Wikipedia (+ WikiMed + streetzim).

## 9. Open questions

1. **Should `compare_articles` pre-summarize each side before
   comparing, or dump both overviews into context?** The first is
   cheaper (smaller context, two summarization passes); the second is
   closer to what the user hears today. Test both.
2. **How do we detect context reuse vs re-retrieval?** A flag in the
   adapter's `search` / `get_article_section` fixture that marks a
   hit as "already fetched this turn" would let the harness
   distinguish.
3. **Is there a better scoring metric for `clarify` turns than "did
   you emit zero tool calls"?** A positive signal — "reply includes a
   bracketed citation to a prior-turn section title" — would be
   stronger. Could parse with a regex.
