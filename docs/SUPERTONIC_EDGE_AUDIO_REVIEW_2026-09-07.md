# Supertonic boundary noise review

The user reports intermittent computer-like static at sentence boundaries. The current adapter trims low-energy padding and directly queues independent PCM buffers. Trimming does not guarantee zero-valued endpoints, so joining those buffers can introduce a discontinuity and audible click. This is a plausible source, not an acoustically confirmed diagnosis of every reported sound.

After level scaling, apply a raised-cosine fade to the outer 5 milliseconds of each Supertonic PCM buffer. Endpoints become zero; interior samples, buffer duration, sample rate, and queue timing remain unchanged. Processing mutates the existing output buffer without an additional PCM allocation. Kokoro and system speech are unchanged.

Adversarial self-review considered short buffers, overlapping fades, silent audio, invalid sample rates, gain increases, and timing changes. Three regression tests verify zero endpoints across a join, exact preservation of interior samples, non-increasing amplitude, unchanged sample count, and edge cases. Full Swift suite: 657 executed, one skipped, zero failures (656 runnable tests passed).

This targets edge clicks only. A short edge fade cannot remove model-generated noise farther inside the waveform or Bluetooth transport artifacts. Confirmation requires listening to the updated build; no claim is made that the intermittent reported sound was reproduced or fully eliminated.
