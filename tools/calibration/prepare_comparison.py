"""Freeze the existing held-out app suites and local artifacts before evaluation.

No model execution, cloud resources, or quality scoring occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def artifact(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Artifact changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest}


def prepare(args):
    candidate = getattr(args, "candidate", None)
    candidate_manifest = getattr(args, "candidate_manifest", None)
    if (candidate is None) != (candidate_manifest is None):
        raise ValueError("--candidate and --candidate-manifest must be supplied together")
    if candidate is None and getattr(args, "candidate_model_id", None) is not None:
        raise ValueError("--candidate-model-id requires --candidate")
    suite = {"schema_version": 1, "name": "Zimfo frozen held-out comparison v1",
             "description": "Existing regression suites; closed-loop app comparison, excluded from calibration.",
             "conversations": []}
    source_suites, seen, prompts = [], set(), set()
    calibration = json.loads(Path(args.calibration_suite).read_text())
    calibration_ids = {c["id"] for c in calibration["conversations"]}
    calibration_prompts = {t["user"].strip().casefold() for c in calibration["conversations"] for t in c["turns"]}
    for path in args.suite:
        source = json.loads(Path(path).read_text())
        for conversation in source["conversations"]:
            identity = conversation["id"]
            if identity in seen or identity in calibration_ids:
                raise ValueError(f"Duplicate or calibration conversation: {identity}")
            for turn in conversation["turns"]:
                prompt = turn["user"].strip().casefold()
                if prompt in calibration_prompts:
                    raise ValueError(f"Exact calibration prompt overlaps held-out suite: {identity}")
                prompts.add(prompt)
            seen.add(identity)
            suite["conversations"].append(conversation)
        source_suites.append(artifact(path))
    if not seen:
        raise ValueError("Empty held-out suite")
    models = {
        "reference": {"artifact": str(Path(args.reference).resolve(strict=True)), "runtime": "mlx",
                      "capture_model_id": "qwen3.8-27b-bf16", "manifest": artifact(args.reference_manifest)},
        "bonsai": {"artifact": str(Path(args.bonsai).resolve(strict=True)), "runtime": "llamacpp",
                   "capture_model_id": "bonsai-27b-q1-gguf", "manifest": artifact(args.bonsai_manifest)}}
    if candidate is not None:
        # The evaluator derives *both* the chat template and the llama context/KV policy
        # from the lowercased GGUF path (ProbeE2ECLI.swift:733,737-745,818-819): the
        # 16384 / Q4_0 policy that verify_like_for_like requires of a like-for-like arm
        # holds only for a path on the bonsai branch and not the ternary one. A
        # "qwen"-named candidate would capture 32768 / Q8_0 and be refused *after* a
        # 3.8 GB run, so refuse it here instead. This stands in for the real fix, an
        # explicit n_ctx/KV switch in the evaluator.
        artifact_path = str(Path(candidate).resolve(strict=True))
        folded = artifact_path.casefold()
        if "bonsai" not in folded or "ternary" in folded:
            raise ValueError(
                f"--candidate must select bonsai's llama context/KV branch (a path containing "
                f"'bonsai' and not 'ternary'); {artifact_path} would capture the 32768 / Q8_0 "
                f"policy and be refused by verify_like_for_like after the capture")
        models["candidate"] = {
            "artifact": artifact_path, "runtime": "llamacpp", "runtime_parity": "bonsai",
            # ProbeE2ECLI.swift:745-747 records "local-discuss-model" for any GGUF that is
            # neither bonsai nor ternary; override with --candidate-model-id when it differs.
            "capture_model_id": getattr(args, "candidate_model_id", None) or "local-discuss-model",
            "manifest": artifact(candidate_manifest)}
    protocol = {
        "schema_version": 1, "mode": "app-conversation", "suite": suite,
        "source_suites": source_suites,
        "calibration_suite": artifact(args.calibration_suite),
        "sources": {"zim": artifact(args.zim), "streetzim": artifact(args.streetzim)},
        "sampling": {"temperature": 0, "top_p": 1, "top_k": 0, "seed": 42},
        "models": models,
        "evaluator": artifact(args.evaluator),
        "rubric": "Existing binary content/grounding assertions, equal conversations then equal turns; human critical-error review still required",
        "performance_policy": "Sequential runs on the same Mac; cross-runtime memory and timing are diagnostic, not phone forecasts",
        "split_limitations": "Exact scenario/prompt disjointness checked; inspect actual retrieved sources for leakage before activation export",
    }
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "suite.json").write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")
    protocol["merged_suite"] = artifact(root / "suite.json")
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"protocol": str(root / "protocol.json"), "conversations": len(seen),
                      "turns": sum(len(c["turns"]) for c in suite["conversations"]),
                      "protocol_sha256": artifact(root / "protocol.json")["sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", required=True)
    for field in ("calibration-suite", "zim", "streetzim", "reference", "reference-manifest",
                  "bonsai", "bonsai-manifest", "evaluator", "output-dir"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--candidate", help="optional third arm; must share bonsai's llamacpp runtime")
    parser.add_argument("--candidate-manifest", help="required with --candidate")
    parser.add_argument("--candidate-model-id", help="capture model id; default local-discuss-model")
    prepare(parser.parse_args())
