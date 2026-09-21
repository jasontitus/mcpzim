#!/usr/bin/env python3
"""Compare two real, seeded Kokoro builds in separate serial processes.

Both executables must support --output WAV plus its .f32 sidecar. Framework
paths let the same harness load a retained baseline KokoroSwift framework.
Does not play audio, change the app's model, or download weights.
"""
import argparse
import array
import json
import math
import os
from pathlib import Path
import subprocess

CASES = [
    ("punctuation", "af_heart", "Dr. Vladimir V. Putin attended Leningrad State University; later, he joined the KGB in 1975."),
    ("names", "af_heart", "Arnold Alois Schwarzenegger is an Austrian and American actor, businessman, former politician, and former professional bodybuilder."),
    ("british", "bf_emma", "George Bernard Shaw wrote more than sixty plays, including Man and Superman, Pygmalion and Saint Joan."),
    ("long", "af_heart", "Napoleon Bonaparte was a French general and statesman who rose to prominence during the French Revolution and led a series of military campaigns across Europe. He led the French Republic as First Consul, then ruled the French Empire as Emperor of the French. You can ask about his early life, military campaigns, or legacy."),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--baseline-frameworks", required=True)
    parser.add_argument("--candidate-frameworks", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, voice, text in CASES:
        signals = {"first": [], "warm": []}
        for label, frameworks in [("baseline", args.baseline_frameworks),
                                  ("candidate", args.candidate_frameworks)]:
            env = os.environ.copy()
            env.pop("KOKORO_BENCH_CACHE_MB", None)
            env["MCPZIM_KOKORO_MODEL_DIR"] = args.models
            env["DYLD_FRAMEWORK_PATH"] = frameworks
            wav = output / f"{name}-{label}.wav"
            result = subprocess.run(
                [args.executable, "--voice", voice, "--text", text, "--output", str(wav)],
                env=env, capture_output=True, text=True, timeout=180)
            (output / f"{name}-{label}.log").write_text(result.stdout + result.stderr)
            result.check_returncode()
            for phase, suffix in [("first", ".first.wav.f32"), ("warm", ".f32")]:
                signal = array.array("f", Path(str(wav) + suffix).read_bytes())
                assert signal and all(math.isfinite(x) for x in signal), (name, label, phase, "invalid PCM")
                assert max(abs(x) for x in signal) > 0.001, (name, label, phase, "silent PCM")
                signals[phase].append(signal)
        for phase, (before, after) in signals.items():
            assert len(before) == len(after), (name, phase, "duration changed")
            max_error = max(abs(a - b) for a, b in zip(before, after))
            rms_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(before, after)) / len(before))
            row = dict(case=name, phase=phase, voice=voice, samples=len(before),
                       bit_exact=before.tobytes() == after.tobytes(),
                       max_error=max_error, rms_error=rms_error)
            results.append(row)
            print(json.dumps(row), flush=True)
    (output / "parity.json").write_text(json.dumps(results, indent=2) + "\n")
    # No perceptual threshold can prove unchanged quality. Require exact PCM
    # for these scheduling-only changes, rather than silently accepting drift.
    assert all(row["bit_exact"] for row in results), "PCM changed; review before accepting"


if __name__ == "__main__":
    main()
