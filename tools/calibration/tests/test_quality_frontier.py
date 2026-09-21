from copy import deepcopy
import pytest
from quality_frontier import compare


def run(scores, size=100, model="candidate"):
    return {"schema_version": 1, "status": "completed", "protocol_sha256": "a" * 64,
            "mode": "model-only", "model": {"id": model, "revision": "pinned",
            "precision": "BF16" if model == "Qwen/Qwen3.8-27B" else "Q1",
            "source_model": "Qwen/Qwen3.8-27B", "source_revision": "pinned",
            "original_weights": model == "Qwen/Qwen3.8-27B", "deployment_bytes": size,
            "runtime": "test", "machine": "test", "artifact_manifest_sha256": "b" * 64},
            "cases": [{"id": str(i), "conversation_id": str(i), "category": "grounding",
                       "case_key_sha256": "c" * 64, "model_invoked": True,
                       "score": score, "status": "scored", "critical_failures": []}
                      for i, score in enumerate(scores)]}


def test_reports_loss_gain_and_modest_size_increase_without_rejecting():
    result = compare(run([1, 1], model="Qwen/Qwen3.8-27B"), run([0, 1]), [run([0.8, 1], 103)])
    row = result["rows"][2]
    assert row["compression_loss_pp"] == pytest.approx(10)
    assert row["gain_vs_bonsai_pp"] == pytest.approx(40)
    assert row["size_change_vs_bonsai_percent"] == pytest.approx(3)
    assert row["headroom_retained_percent"] == pytest.approx(80)
    assert result["automatic_promotion"] is False
    assert result["rows"][1]["compression_loss_pp"] is None


@pytest.mark.parametrize("mutate", [
    lambda r: r["cases"].pop(),
    lambda r: r["cases"].append(deepcopy(r["cases"][0])),
    lambda r: r.update(protocol_sha256="d" * 64),
    lambda r: r.update(status="running"),
    lambda r: r["cases"][0].update(case_key_sha256="d" * 64),
    lambda r: r["cases"][0].update(score=float("nan")),
    lambda r: r["cases"][0].update(status="skipped"),
    lambda r: r["cases"][0].update(status="failed", score=1),
    lambda r: r["cases"][0].update(model_invoked=False),
    lambda r: r["model"].update(deployment_bytes=0),
    lambda r: r["model"].update(source_revision="different-checkpoint"),
])
def test_unpaired_or_misleading_reports_rejected(mutate):
    reference = run([1, 1], model="Qwen/Qwen3.8-27B")
    candidate = run([1, 1])
    mutate(candidate)
    with pytest.raises(ValueError):
        compare(reference, run([0, 1]), [candidate])


def test_critical_regression_survives_better_aggregate_and_failure_stays_in_denominator():
    reference = run([1, 1, 1], model="Qwen/Qwen3.8-27B")
    candidate = run([0, 1, 1])
    candidate["cases"][0].update(status="failed", critical_failures=["wrong-source"])
    row = compare(reference, run([0, 0, 1]), [candidate])["rows"][2]
    assert row["gain_vs_bonsai_pp"] > 0
    assert row["score"] == pytest.approx(2 / 3)
    assert row["failed_cases"] == ["0"]
    assert row["new_critical_failures_vs_bonsai"] == {"0": ["wrong-source"]}


def test_no_percentage_retention_when_reference_is_not_better():
    row = compare(run([0], model="Qwen/Qwen3.8-27B"), run([1]), [run([0.5])])["rows"][2]
    assert row["headroom_retained_percent"] is None
    assert row["compression_loss_pp"] == -50


def test_reject_dequantized_reference_and_preserve_conversation_weighting():
    reference = run([1, 0, 0], model="Qwen/Qwen3.8-27B")
    reference["cases"][2]["conversation_id"] = "1"
    bonsai = deepcopy(reference)
    bonsai["model"]["id"] = "bonsai"
    assert compare(reference, bonsai, [])["rows"][0]["score"] == 0.5
    reference["model"]["original_weights"] = False
    with pytest.raises(ValueError):
        compare(reference, bonsai, [])
