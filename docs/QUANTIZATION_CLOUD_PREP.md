# CPU-only preparation before the paid GPU run

Updated 2026-09-19. The runnable preparation code is in
`tools/calibration/cloud_prep/`. **No VM has been created by this component yet.**
Planning and unit tests pass; actual guest boot, driver availability, registry
pull and disk restore remain checks to run once input staging and the final
solver image are committed.

## Fixed resources and launch gates

The launcher only supports `n4-standard-2` in `us-central1-b`: two CPU cores,
8 GiB RAM and **no GPU**. The machine specification was read from the project
API. The base OS is pinned to
`deeplearning-platform-release/common-cu129-ubuntu-2204-nvidia-580-v20260909`,
image ID **3612508179781164991**, verified `READY` through the API. It has the
NVIDIA 580 driver expected for the CUDA 13 solver container. The guest will verify
the installed module for its running kernel; a missing first-boot driver is a
preparation failure, not something deferred to the GPU rental.

It creates named Hyperdisk Balanced disks: **100 GiB boot** and **256 GiB input**.
Both have auto-delete disabled and remain independently of the CPU VM. Docker's
downloaded layers and configured toolkit live on the boot disk; verified model and
calibration inputs live on the data disk. Both must be reused to avoid repeating
installs, image downloads or input staging when the GPU starts.

Before any resource creation, the launcher requires:

- A local input manifest matching the successful GCS staging receipt's exact
  SHA-256, byte length, bucket and positive generation. The remote committed bytes
  are checked again before execution.
- The final solver image in the private Zimfo registry, addressed as
  `image@sha256:...`, and a successful read-only registry lookup. Tags are rejected.
- Unchanged copies of the reviewed startup/config/bootstrap files.
- The exact expected OS image ID.

The guest probes imports of `solver.run`, **`solver.job`**, `restore_inputs` and
Google Storage inside the pinned image. It checks the PyTorch CUDA build is 13.x
and no CUDA device is available on this CPU machine. A utility-only or incomplete
image therefore does not satisfy preparation readiness.

## Network, permissions, time and cost

The VM uses the dedicated
`zimfo-quant-worker@tiltastech-zimfo.iam.gserviceaccount.com` identity with existing
storage read/create and registry read roles. It uses metadata credentials, with
no service-account key file. Registry login credentials exist only under `/run`
and are removed after the pull; they are not baked into the retained boot disk.

An ephemeral IPv4 address permits outbound registry/API and, if needed, package
downloads during CPU preparation. A dedicated priority-zero deny-all ingress rule
targets only `zimfo-cpu-prep`; no SSH rule or public app port is opened. Existing
firewall rules with that name but different properties are rejected, not modified.
SSH is also disabled inside the guest. Progress is available from serial output,
retained local logs, and one final GCS report. No per-step status objects are sent.

The Compute Engine API has a **one-hour maximum runtime and DELETE action**. A
guest systemd timer also requests shutdown at 55 minutes, and bootstrap is bounded
to 50 minutes. Successful or failed bootstrap publishes its final report and shuts
down early. Interrupted or uncertain create responses do not trigger a new VM.

The published Iowa on-demand N4 CPU rate is **$0.0907/hour** for this size; the
one-hour VM component is about nine cents. Storage, IPv4, registry/GCS operations
and subsequent images/snapshots are additional. **The retained 356 GiB of disks
continue incurring storage charges after the runtime cap or VM deletion.** The
runtime guard is a compute limit, not a total billing cap. Sources:
[Google CPU pricing](https://cloud.google.com/products/compute/pricing/general-purpose),
[maximum VM runtime](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime),
[N4 specifications](https://docs.cloud.google.com/compute/docs/general-purpose-machines#n4_series).

## Commands and execution order

Create a local reviewable plan only after both inputs and image exist:

```sh
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python \
  tools/calibration/cloud_prep/prepare.py plan \
  --input-status /absolute/path/to/staging/status.json \
  --input-manifest /absolute/path/to/staging/inputs-manifest.json \
  --image us-central1-docker.pkg.dev/tiltastech-zimfo/zimfo-quantization/solver@sha256:ACTUAL_DIGEST \
  --output /absolute/path/to/new/preparation-plan
```

The generated `plan.json` contains the exact disk and VM command arrays. After
review, `prepare.py execute PLAN_DIRECTORY` rechecks everything and creates the
dedicated firewall if absent, the two named disks, then the CPU VM. It never creates
a GPU. If a create request fails or has an uncertain response, inspect the retained
plan and named resources; do not generate another plan as an automatic retry.

Inside the guest, the reviewed bootstrap hash is checked before Python execution.
The guest confirms its CPU machine type and the data device is a 256 GiB block
device with **no existing filesystem signature** before formatting it. It configures
Docker/NVIDIA Container Toolkit, installing missing host utilities on CPU time if
the pinned image needs them. It does not upgrade/rebuild the NVIDIA driver.
Unsupported or missing driver/toolkit components fail here. Once utilities are
ready, installed automatic apt/unattended-update units are masked so the later
GPU boot cannot silently install packages or change the reviewed kernel.

Next it pulls the exact image digest, verifies its architecture and imports, then
downloads the generation-pinned input manifest. It runs `python -m restore_inputs`
inside the container with the expected manifest SHA-256. That consumer validates
every downloaded object, extracted calibration tensor and original model file
before publishing the prepared input directory. The guest reports its kernel,
driver, toolkit, container identity and full restore validation, then syncs disks.

The final GCS object is `preparation/RUN_ID/result.json` and is created once using
a create-only generation condition. Logs are retained at `/var/log/zimfo-prep.log`
on the boot disk. `prepared_cpu_only` explicitly leaves `gpu_smoke_passed: false`.

## Hand-off to the GPU launcher

After a success report and guest shutdown, run:

```sh
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python \
  tools/calibration/cloud_prep/prepare.py finish PLAN_DIRECTORY
```

This checks the report and exact ownership label before deleting **only this
stopped CPU VM** with `--keep-disks=all`. If the one-hour guard already deleted it,
it verifies that state instead. It then requires both disks to be ready, detached,
correctly named/labelled and correctly sized. Nothing deletes the disks, GCS input
objects or checkpoints.

The resulting `ready.json` is the GPU launch contract:

- `boot_disk` and `data_disk`: names, immutable IDs, self links, size and type.
- `project`, `zone`, `source_image_id`, `config_sha256` and preparation report
  generation.
- `production_image`: exact registry digest; `input_commit`: pinned object,
  generation, SHA-256 and bytes; `bucket`.
- `runtime_proof`: kernel, NVIDIA driver/module, toolkit and container probe;
  `restore_validation` records the independent input validation.
- `paths`: host mount `/mnt/zimfo-inputs`, model
  `/mnt/zimfo-inputs/prepared/model`, calibration
  `/mnt/zimfo-inputs/prepared/calibration`, container workdir
  `/opt/zimfo-runtime`, data device `/dev/disk/by-id/google-zimfo-inputs`.

The GPU VM can attach these exact detached disks directly, with auto-delete false,
and mount the data device at the same path. It must replace the CPU startup metadata
with its own load/check/smoke/train startup. It must not run the CPU disk-formatting
bootstrap on prepared inputs. An optional `snapshot-recipe PLAN_DIRECTORY` prints
commands for an input-disk snapshot and prepared boot image, for later clones; it
does not execute them or delete resources.

## Explicit storage performance and cost

Disk performance is explicitly provisioned: boot 3,000 IOPS / 140 MiB/s;
data 3,000 IOPS / 750 MiB/s. The latter supports faster full-model loads on the
GPU host without paying for extra IOPS. Actual throughput is also limited by
the attached machine. Defaults would silently provision additional billable
performance, so they are not used.

At the September 19 Iowa prices, the 356 GiB capacity costs approximately
$0.0390/hour and the additional 610 MiB/s costs $0.0334/hour: about **$0.0724/hour
while retained**, or $1.74/day. These charges continue after either VM is deleted.
GCS/registry storage and transfer are separate. This is a calculated estimate
from [Google's disk pricing](https://cloud.google.com/compute/disks-image-pricing)
and [Hyperdisk performance limits](https://docs.cloud.google.com/compute/docs/disks/hd-types/hyperdisk-balanced).
Retain the disks for the experiment/recovery; review storage retention when the
run completes rather than leaving them indefinitely without accounting for cost.

## Verification and adversarial findings

Local tests:

```sh
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python \
  -m pytest tools/calibration/cloud_prep/tests -q
bash -n tools/calibration/cloud_prep/startup.sh
```

**23 tests passed** and shell syntax passed. Tests cover no-cloud planning, missing
readiness artifacts, immutable image requirements, tampered startup/config/commands,
GPU flag rejection, firewall ownership, authentication failures, detached-disk
receipts, retained-disk policy, wrong CPU type, and the startup command sequence.
They do not claim an actual CPU guest or prepared disk has passed yet.

Independent review caught two startup issues that were fixed: invoking the restorer
by file path would miss its package location in the image, so it now uses
`python -m restore_inputs`; the downloaded bootstrap is now hash checked before it
runs. The hash check is compatible with the base image's Python 3.10. A remaining
real-environment gate is whether this DLVM image has its NVIDIA kernel module
available without a GPU on first boot. If not, preparation stops on the cheap CPU
machine, retaining logs and disks for diagnosis.

## Recovering a definite CPU stockout without recreating disks

The first N4 standard2 create attempt returned a completed
`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` operation. The disks were retained;
there was no running VM. `recover` performs read-only cloud inspection and
writes a **new** local plan, including byte-for-byte copies of the previous
plan/config/startup code. It requires exact disk IDs, owner labels, sizes,
Hyperdisk performance settings, pinned boot image ID, no disk users, VM absence,
and a DONE stockout operation for that exact instance. Attach/detach timestamps
must fall within the failed create operation. The input guest still refuses to
format a disk containing an existing filesystem signature.

```sh
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python \
  tools/calibration/cloud_prep/prepare.py recover \
  tools/calibration/runs/cloud-cpu-prep-20260919-v2 \
  --output tools/calibration/runs/cloud-cpu-prep-20260919-v3 \
  --machine-type n4-standard-4 \
  --boot-disk-id 6809116840565058866 \
  --input-disk-id 6810041422469836046 \
  --image us-central1-docker.pkg.dev/tiltastech-zimfo/zimfo-quantization/runtime@sha256:0c95864edb0b19dd9c722c5e2cca1ca210fab693b2f142749424711dc2956ab9
```

This command only prepares the recovery plan; it does not launch. After review,
`execute` rechecks the retained disk identities and attachment history, verifies
the final image and committed input manifest, and creates only the CPU VM.
It creates no new disks. Only N4 standard2, N4 standard4 and C4 standard2 are permitted; standard4
has a recorded CPU rate of $0.1814/hour, with the same one-hour API deletion cap.
Disk, IPv4 and service costs remain separate. Captured cloud error messages are
now included in failures so a definite stockout is distinguishable from an
uncertain operation. An uncertain create must still be inspected, never retried
blindly.

A second definite N4 standard4 stockout can be recovered with the same command
and disk IDs using `--machine-type c4-standard-2` and a new local output
directory. This additional CPU-only shape is explicitly allowlisted in both the
controller and guest; its recorded Iowa CPU rate is $0.096866/hour. The original
source plan stays unchanged, and recovery validates the latest failed create
operation and current disk attachment timestamps before generating commands.

## September19 continuation of the first live restore

Monitoring measured22–25MB/s during the original sequential8MiB ranged reads,
which would exceed the40-minute restore-command limit for79.1GB. Before timeout,
`cloud_prep.resume_cpu` stopped the owned C4 VM, preserved both disks, replaced
only the CPU restore entry point with a hash-bound read-only source mount, and
restarted the same CPU VM. The final production solver image is unchanged.
The v6 frozen plan records this attempt and uses a new immutable result key.

The continuation explicitly adopts the stopped legacy staging tree, verifies
metadata→sequence→file hashes, reuses completed archives/model shards, repairs
only manifest-derived partial paths, and rejects symlinks or unknown files.
A lock and manifest/destination ownership record bind future resumes.64MiB
GCS reads amortize round-trip cost; progress is serial/local only. The new guest
also verifies the NVIDIA open-module license required for Blackwell.

Restore tests cover complete publication after revalidation, known partial
repair, avoiding downloads of completed content, wrong manifests, foreign files,
symlinks, corrupt completed weights, and missing explicit legacy adoption.
The CPU/GPU bootstrap and restore suites passed74 tests after these changes.
The additional CPU attempt retains a one-hour API limit and55-minute guest
shutdown. It does not authorize an additional GPU run or extend its limit.
