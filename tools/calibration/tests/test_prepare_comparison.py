import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prepare_comparison import artifact, prepare


def settings(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return str(path)
    suite = write("heldout.json", {"conversations": [{"id": "evaluation", "turns": [{"user": "Where is Rome?"}]}]})
    cal = write("calibration.json", {"conversations": [{"id": "calibration", "turns": [{"user": "Explain coral reefs"}]}]})
    data = write("data.json", {"data": "fixture"})
    return SimpleNamespace(suite=[suite], calibration_suite=cal, zim=data, streetzim=data,
                           reference=data, reference_manifest=data, bonsai=data,
                           bonsai_manifest=data, evaluator=data, output_dir=str(tmp_path / "frozen"))


def test_freezes_suite_sources_and_rejects_overwriting(tmp_path):
    args = settings(tmp_path)
    prepare(args)
    protocol = json.loads((tmp_path / "frozen/protocol.json").read_text())
    assert protocol["suite"] == json.loads((tmp_path / "frozen/suite.json").read_text())
    assert protocol["sources"]["zim"] == artifact(args.zim)
    assert protocol["models"]["bonsai"]["runtime"] == "llamacpp"
    with pytest.raises(FileExistsError):
        prepare(args)


def with_candidate(args, tmp_path, name="bonsai-candidate-q1.gguf"):
    def write(field, value):
        path = tmp_path / field
        path.write_text(json.dumps(value))
        return str(path)
    args.candidate = write(name, {"candidate": "gguf fixture"})
    args.candidate_manifest = write("candidate-manifest.json",
                                    {"status": "runtime-verified", "artifact_sha256": "0" * 64})
    args.candidate_model_id = None
    return args


def test_without_candidate_the_two_role_protocol_is_unchanged(tmp_path):
    two_role = settings(tmp_path)
    prepare(two_role)
    plain = json.loads((tmp_path / "frozen/protocol.json").read_text())
    assert list(plain["models"]) == ["reference", "bonsai"]
    three_role = with_candidate(settings(tmp_path), tmp_path)
    three_role.output_dir = str(tmp_path / "frozen-three-role")
    prepare(three_role)
    extended = json.loads((tmp_path / "frozen-three-role/protocol.json").read_text())
    assert list(extended["models"]) == ["reference", "bonsai", "candidate"]
    # Only the model map and its own merged-suite copy may differ from the two-role freeze.
    assert {key: value for key, value in extended.items() if key not in ("models", "merged_suite")} == \
           {key: value for key, value in plain.items() if key not in ("models", "merged_suite")}
    assert extended["models"]["reference"] == plain["models"]["reference"]
    assert extended["models"]["bonsai"] == plain["models"]["bonsai"]


def test_candidate_shares_bonsai_runtime_and_sampler(tmp_path):
    args = with_candidate(settings(tmp_path), tmp_path)
    prepare(args)
    protocol = json.loads((tmp_path / "frozen/protocol.json").read_text())
    candidate = protocol["models"]["candidate"]
    assert candidate["runtime"] == protocol["models"]["bonsai"]["runtime"] == "llamacpp"
    assert candidate["runtime_parity"] == "bonsai"
    assert candidate["capture_model_id"] == "local-discuss-model"
    assert candidate["artifact"] == str(Path(args.candidate).resolve())
    assert candidate["manifest"] == artifact(args.candidate_manifest)
    assert protocol["sampling"] == {"temperature": 0, "top_p": 1, "top_k": 0, "seed": 42}


def test_candidate_model_id_override_and_path_selection(tmp_path):
    args = with_candidate(settings(tmp_path), tmp_path)
    args.candidate_model_id = "qwen3.8-27b-q1-rtn"
    prepare(args)
    assert json.loads((tmp_path / "frozen/protocol.json").read_text())["models"]["candidate"]["capture_model_id"] \
        == "qwen3.8-27b-q1-rtn"
    missing = with_candidate(settings(tmp_path), tmp_path, name="qwen-candidate.gguf")
    missing.candidate_manifest = None
    missing.output_dir = str(tmp_path / "frozen-missing")
    with pytest.raises(ValueError, match="together"):
        prepare(missing)
    blocked = with_candidate(settings(tmp_path), tmp_path, name="candidate-q1.gguf")
    blocked.output_dir = str(tmp_path / "frozen-blocked")
    with pytest.raises(ValueError, match="context/KV branch"):
        prepare(blocked)
    # A "qwen"-named candidate must be refused at freeze time rather than after the
    # capture: the evaluator derives the llama context/KV policy from the lowercased path
    # (ProbeE2ECLI.swift:733,818-819), so it would record 32768 / Q8_0 and
    # verify_like_for_like would reject it - at the cost of a 3.8 GB capture run.
    qwen_named = with_candidate(settings(tmp_path), tmp_path, name="qwen3.8-27b-q1-rtn.gguf")
    qwen_named.output_dir = str(tmp_path / "frozen-qwen-named")
    with pytest.raises(ValueError, match="context/KV branch"):
        prepare(qwen_named)


def test_rejects_duplicate_conversation_before_writing(tmp_path):
    args = settings(tmp_path)
    args.suite *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(args)
    assert not (tmp_path / "frozen").exists()


def test_rejects_calibration_prompt_overlap_before_writing(tmp_path):
    args = settings(tmp_path)
    from pathlib import Path
    Path(args.suite[0]).write_text(json.dumps({"conversations": [
        {"id": "evaluation", "turns": [{"user": "  EXPLAIN CORAL REEFS "}]}]}))
    with pytest.raises(ValueError, match="overlaps"):
        prepare(args)
    assert not (tmp_path / "frozen").exists()
