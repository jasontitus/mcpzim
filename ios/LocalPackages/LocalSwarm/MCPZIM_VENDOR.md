# Vendored LocalSwarmEngine

Vendored from https://github.com/jasontitus/localswarm at commit
`8febd512756f7a9b43972eb93419ce4222309269`, verified against upstream `main` on 2026-09-05 (engine `Sources/` + `Tests/` +
`Package.swift` only — the upstream repo's own SwiftUI app, Android app,
and Go peer are not vendored). That commit includes the DS4
perf-review fixes and checkpointed-persistence hardening (via upstream's
`claude/ds4-branches-review-zutieh`), the reconciled folder-share design
(single folder = Go-conformant unprefixed folder swarm), and the
mixed-share expansion Zimfo relies on (`Chunker.mixedSources` — files +
folders in one swarm, each folder prefixed with its name).

LocalSwarmEngine is the local-first peer-to-peer file swarming engine Zimfo
uses for **Nearby Sharing**: seeding your ZIM library to nearby devices over
AWDL / peer-to-peer Wi-Fi (TCP+TLS and QUIC transports), and pulling a
friend's ZIMs from every nearby source in parallel with SHA-256 per-chunk
verification and resume.

Local path (not a git URL) for the same reasons as the other vendored
packages here: hermetic builds with no auth against a private repo, and the
app pin moves only when this directory is deliberately re-synced.

## Re-syncing with upstream

From the mcpzim repository root, fetch and compare the sibling checkout:

```sh
git -C ../localswarm fetch origin
git -C ../localswarm status --short --branch
git -C ../localswarm log 8febd512756f7a9b43972eb93419ce4222309269..HEAD -- Sources Tests Package.swift
diff -ru ../localswarm/Sources ios/LocalPackages/LocalSwarm/Sources
diff -ru ../localswarm/Tests ios/LocalPackages/LocalSwarm/Tests
```

Reconcile new upstream changes against the recorded revision and the local
patches below. Preserve `LICENSE` and `NOTICE`; update the revision only after
reviewing the source differences. Do not delete and blindly replace Sources
or Tests: that would discard integration fixes. Run the package suite and the
signed app integration checks documented in the update report.

Local patches are retained; do not overwrite them on the next sync. The
previous note incorrectly claimed there were none. Upstream changes between
`6d6d47e` and this pin are review documents only; all upstream engine changes
were already present. The following pre-existing differences are preserved:

- `AuthToken.swift`: use the shared hex encoder.
- `ManifestCache.swift` and its tests: bound the cache to 32 entries.
- `SwarmManager.swift`: preserve PIN on receive/resume, clean download metadata
  on failure/cancel, and avoid repeated chunk-layout scans in benchmarks.
- `TransferLogger.swift`: reuse the ISO formatter on its owning queue.

Added during this integration pass:

- `Manifest.swift`: overflow-safe untrusted manifest size/range validation.
- `SwarmManager.swift`: startup/receive attempt ownership, canceled hosting
  preparation checks, stale snapshot rejection, and ordered disk cleanup.
- `SwarmSession.swift`: terminal Stop state, drain queued disk writes before
  flushing, and prevent late completion from re-advertising a stopped swarm.
- `ChunkStore.swift`: consistent existing-ancestor path resolution on iOS;
  reject escaping or dangling data/metadata symlinks.
- `SecurityTests.swift` and `ManagerLifecycleTests.swift`: regression coverage.

Integration evidence and adversarial review are documented in
[`docs/LOCALSWARM_UPDATE_2026-09-05.md`](../../../docs/LOCALSWARM_UPDATE_2026-09-05.md).
On 2026-09-05 all patches above, including the earlier cache/PIN/performance
fixes and their tests, were ported into the sibling `../localswarm` checkout's
`main` working tree. The port review also moved CSV formatting onto the logger's
serial queue in both copies, making formatter reuse actually queue-confined.
The sibling's 58 package tests pass, and its complete `Sources/`, `Tests/`, and
`Package.swift` match this copy byte for byte. This is a local, uncommitted port;
the upstream revision recorded above remains unchanged until publication.

## License

Apache-2.0 (see `LICENSE` / `NOTICE` in this directory). The rest of the
mcpzim repo is MIT; this directory keeps its upstream license.
