# MCPZim FluidAudio fork

Upstream: <https://github.com/FluidInference/FluidAudio>

Base version: `0.15.5` (`19600a48`)

This local package preserves FluidAudio's source, tests, CLI, license, and
notices. MCPZim carries one intentional source change:

- `Supertonic3Constants.maxChunkLengthLatin` is 96 instead of upstream's 70.

FluidAudio issue #669 measured much better aggregate word accuracy at 70 than
at 110, including 17.6% WER for one 105-character phrase. However, issue #736
and MCPZim's iPhone 17 Pro Max listening tests found that 70-character chunks
produce obvious sentence restarts during paragraph narration. MCPZim therefore
uses 96 as a compromise, with app-level boundary-aware silence trimming and a
94-character dispatch ceiling. Keep the dispatch ceiling two characters below
the library limit.

Before updating the upstream base, rerun:

1. FluidAudio's `Supertonic3TypesTests` and `Supertonic3TextChunkerTests`.
2. MCPZimKit's `StreamingSpeechPolicyTests`.
3. The physical-device Washington acceptance passage in `RootView` using
   `MCPZIM_BENCH_SUPERTONIC=1`, checking continuity and every spoken word.
