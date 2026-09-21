"""Small live GCS checkpoint proof. Writes only a fresh tests/checkpoints-* prefix.

Keeps evidence objects; never creates buckets/VMs or deletes unrelated state.
This is a transport and real CPU optimizer resume test, not full-model CUDA proof.
"""
import argparse
import json
from pathlib import Path
import random
import tempfile
import time
import uuid

import numpy as np
import torch
from google.api_core.exceptions import NotFound, PreconditionFailed

from checkpoints import CheckpointError, capture_rng_state, restore_rng_state
from gcs_checkpoints import GCSCheckpointStore, make_client


def run(bucket, project, use_gcloud):
    prefix = "tests/checkpoints-" + uuid.uuid4().hex
    store = GCSCheckpointStore(bucket, prefix, client=make_client(project, use_gcloud=use_gcloud))
    identity = {"baseline_repo": "transport-test-only", "baseline_revision": "a" * 40,
                "solver_revision": "b" * 40, "calibration_sha256": "1" * 64,
                "solver_config_sha256": "2" * 64, "candidate_database_sha256": "3" * 64,
                "runtime_sha256": "4" * 64}
    report = {"status": "running", "bucket": bucket, "prefix": prefix,
              "scope": "live transport and tiny CPU optimizer; not full Qwen or CUDA", "checks": {}}
    with tempfile.TemporaryDirectory(prefix="checkpoint-live-") as directory:
        root = Path(directory)
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        parameter = torch.nn.Parameter(torch.randn(4))
        optimizer = torch.optim.Adam([parameter], lr=.05)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)

        def update(param, opt, sched):
            target = torch.rand_like(param) + random.random() + float(np.random.random())
            opt.zero_grad()
            (param - target).square().mean().backward()
            opt.step()
            sched.step()
            return target

        for _ in range(3):
            update(parameter, optimizer, scheduler)
        state = {"solver": {"parameter": parameter.detach()}, "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": capture_rng_state(),
                 "progress": {"completed_updates": 3, "next_update": 3}}
        payloads = {}
        for role, value in state.items():
            path = root / role
            torch.save(value, path)
            payloads[role] = path
        receipt = store.publish("step-3", identity, payloads)
        report["checks"]["publish_verified"] = True
        assert store.publish("step-3", identity, payloads) == receipt
        report["checks"]["idempotent_repeat"] = True
        expected_target = update(parameter, optimizer, scheduler)
        expected_parameter = parameter.detach().clone()
        expected_moment = optimizer.state[parameter]["exp_avg"].clone()
        store.restore("step-3", identity, root / "resume", commit_generation=receipt["commit"]["generation"])
        load = lambda name: torch.load(root / "resume" / name, weights_only=True)
        resumed = torch.nn.Parameter(load("solver")["parameter"])
        opt2 = torch.optim.Adam([resumed], lr=.05)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=2, gamma=.8)
        opt2.load_state_dict(load("optimizer"))
        sched2.load_state_dict(load("scheduler"))
        restore_rng_state(load("rng"))
        actual_target = update(resumed, opt2, sched2)
        assert torch.equal(expected_target, actual_target)
        assert torch.equal(expected_parameter, resumed)
        assert torch.equal(expected_moment, opt2.state[resumed]["exp_avg"])
        assert scheduler.state_dict() == sched2.state_dict()
        report["checks"]["next_adam_update_rng_scheduler_exact"] = True

        try:
            store.restore("step-3", identity, root / "wrong-generation", commit_generation=receipt["commit"]["generation"] + 1)
        except (NotFound, PreconditionFailed, CheckpointError):
            report["checks"]["wrong_commit_generation_rejected"] = True
        else:
            raise AssertionError("Wrong generation was accepted")
        try:
            store.restore("step-3", {**identity, "calibration_sha256": "e" * 64}, root / "wrong-identity")
        except CheckpointError:
            report["checks"]["wrong_identity_rejected"] = True
        else:
            raise AssertionError("Wrong identity was accepted")

        original_create = store.backend.create

        def interrupted(key, *args):
            if key.endswith("/commits/interrupted.json"):
                raise ConnectionError("injected process loss before commit")
            return original_create(key, *args)

        store.backend.create = interrupted
        try:
            store.publish("interrupted", identity, payloads)
        except ConnectionError:
            assert "interrupted" not in store.committed_snapshots()
            report["checks"]["interrupted_publish_has_no_commit"] = True
        finally:
            store.backend.create = original_create
        retry_receipt = store.publish("interrupted", identity, payloads)
        store.verify("interrupted", identity, commit_generation=retry_receipt["commit"]["generation"])
        report["checks"]["interrupted_publish_retry_recovers"] = True

        payloads["progress"].write_bytes(b"conflicting state")
        try:
            store.publish("step-3", identity, payloads)
        except CheckpointError:
            report["checks"]["conflicting_commit_rejected"] = True
        else:
            raise AssertionError("Conflicting commit was accepted")
        store.verify("step-3", identity, commit_generation=receipt["commit"]["generation"])
        report["checks"]["original_survives_conflict"] = True
        objects = list(store.backend.client.list_blobs(bucket, prefix=prefix + "/"))
        report.update(status="passed", commit=receipt["commit"], object_count=len(objects),
                      stored_bytes=sum(blob.size for blob in objects),
                      retained_for_evidence=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--project", default="tiltastech-zimfo")
    parser.add_argument("--gcloud", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = run(args.bucket, args.project, args.gcloud)
    report["elapsed_seconds"] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
