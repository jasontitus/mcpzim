# LocalSwarm update and integration review — September 5, 2026

## Revision and preserved work

Fetched `origin` in the clean sibling `../localswarm` checkout. Both its `HEAD`
and `origin/main` are `8febd512756f7a9b43972eb93419ce4222309269`. Updated the
vendored revision from `6d6d47e`. The upstream commits between those revisions
add review documents, not engine code; this is not a new upstream transport
or throughput improvement.

The app already contained four locally modified engine files and a cache
regression test, despite the vendor note saying there were no patches. Those
changes are preserved: PIN forwarding/resume, download metadata cleanup,
bounded manifest caching, shared hex encoding, reused date formatting, and
benchmark layout caching. Package manifest, license, and notice match the
sibling checkout. The initial update did not modify sibling sources. The
subsequent requested port is documented below; nothing was pushed.

The [vendor record](../ios/LocalPackages/LocalSwarm/MCPZIM_VENDOR.md) now lists
all patches and replaces the destructive re-copy instructions with a comparison
and reconciliation workflow.

## Port back to the main LocalSwarm checkout

At the user's request, copied the reviewed patches into the clean sibling
`../localswarm` checkout on branch `main`, based on `8febd512756f7a9b43972eb93419ce4222309269`.
Seven engine files and three test files carry all integration fixes, including
the pre-existing PIN, cache, encoding, formatting, and benchmark improvements.
The standalone SwiftUI app, Android app, Go peer, package manifest, and license
files were not changed.

The port's adversarial self-review rechecked attempt ownership, queue ordering,
session-specific disk draining, integer bounds, destination containment, and
public API call compatibility. It found that the reused `ISO8601DateFormatter`
was still invoked before dispatching onto the logger's serial queue. Both
copies now format the CSV inside that queue, retaining the event's original
timestamp. This removes concurrent access to the shared formatter and keeps
formatting work off the caller's queue.

Validation in the sibling checkout: `swift test` passed **58 tests**, zero
failures, including real local TCP/QUIC tests and the added lifecycle/security
regressions. `git diff --check` passed. All 31 source/test files and
`Package.swift`, `LICENSE`, and `NOTICE` match the vendored copy byte for byte.
The test log is `/private/tmp/zimfo-localswarm-upstream-tests.log`; the sibling
also contains `MCPZIM_INTEGRATION_2026-09-05.md` with the patch inventory.
No further phone run was needed for the port; the physical-phone evidence
below applies to the preceding integration run, before the logger queue fix.
Changes remain local and uncommitted in both repositories.

## Findings and repairs

| ID | Severity | Finding | Repair and evidence |
|---|---|---|---|
| LS1 | P1 | Peer-controlled `Int64.max` file sizes overflow manifest validation arithmetic and can terminate the app. | Use checked totals and quotient/remainder for chunk counts; bound ranges by actual hashes before integer conversion. Public download entry also validates manifest identity and selections. Maximum sizes/indices and public-boundary rejection tests pass. |
| LS2 | P1 | Pause before startup registration is ignored; late download callbacks can outlive Stop; replacing a session can call its queue-confined `stop()` on the main queue. Concurrent cleanup can race new startup. | Identify each start/receive attempt, reject retired callbacks, register before starting networking, honor pause during creation, and order stop/cleanup on the network queue. Session Stop is terminal, drains only its own disk work before checkpointing, and prevents late completion from re-advertising. Lifecycle tests and PIN-preserving phone pause/resume pass. |
| LS3 | P1 | Stop Sharing while hashing does not cancel the pending preparation; it can still publish the share after Stop. Old snapshots can restore a removed row. | Clear pending preparations and verify ownership before advertising; accept host snapshots only from the current session. The regression fails against the previous vendored code and passes after repair. |
| LS4 | P1 | Physical-phone receive fails with `ChunkStore.StoreError.unsafePath`: Foundation normalizes an existing iOS destination and a missing child inconsistently. The old lexical check also does not enforce its promised symlink containment. | Resolve each existing ancestor before appending missing components; require data and metadata destinations to remain inside the store; reject dangling links. Directory-alias and escaping data/sidecar symlink tests pass. The formerly failing phone transfer now completes over both transports. |

The existing upstream automated review was treated as leads, not accepted as
a verified audit. LS1 was confirmed in the actual source. This pass reviews
the changed integration boundaries; it does not claim to resolve every item in
that separate review or audit the Go/Android applications.

## Adversarial verification

- Initial vendored suite: **49 tests passed**.
- Final engine suite: **58 tests passed**, zero failures. Includes production
  QUIC handshake/round trip, TCP/QUIC hosting, chunk corruption rejection,
  persisted-bitfield resume, folder layout, cross-language conformance vectors,
  cache bounds, and the new lifecycle/path/numeric regressions.
- Copied the previous committed package to an isolated temporary directory and
  ran the new lifecycle tests against it. Pause-during-startup and
  Stop-while-hashing both failed (five failed assertions across two tests).
  The cancellation test was also run and passed on that baseline; it is not
  presented as proof that the old code failed that particular assertion.
- App integration: **9 ModelSharingTests passed** in the signed Mac test host:
  GGUF filename/size adoption, duplicate/truncated-cache behavior, and known
  voice-folder path routing. These tests do not synthesize or play audio.
- Signed iPhone Debug build, signed Mac Release test build, and both repository
  signature gates passed. The iPhone build is installed.

The opt-in DEBUG phone probe (`MCPZIM_SWARM_PROBE=1`) uses two independent
`SwarmManager` instances in the actual app. It advertises only generated
fixtures under a temporary directory, discovers its own content-addressed
swarm through Bonjour, and never shares the user's library/model files.
The PIN is random and not written to the report. Six checks passed:

1. Mixed flat file plus nested folder manifest preserves all paths.
2. Incorrect PIN is rejected by the real seeder.
3. QUIC receive completes with every output byte equal to its fixture.
4. Pause during queued startup is immediately reflected as paused.
5. TCP resume retains the correct PIN and completes with identical bytes.
6. Successful fixture directory is removed.

Both transfers contained three files totaling **3,235,861 bytes**. Observed
elapsed times were QUIC **0.316 seconds** and TCP **0.094 seconds**. These are
same-device integration runs through Network.framework; they are **not**
AWDL radio throughput benchmarks or a Mac-to-phone transfer measurement.

The probe initially failed. Improved error reporting exposed the exact store
failure, then the path fix changed that failing transfer into a pass. Tests
require a completion callback and compare actual output bytes; mere discovery,
a progress row, or a timeout does not qualify as success.

The adversarial self-review also challenged the new shutdown code. A shared
I/O queue barrier could make Stop wait for a different share's multi-gigabyte
hashing job. Shutdown now waits on a group containing only that session's
reads/writes. A regression holds unrelated I/O indefinitely and requires Stop
to complete within one second. Late network callbacks also check the terminal
Stop state before scheduling more work.

## Limits and reproducibility

No audio, microphone, Siri, or real voice-model loading was used. No TestFlight
upload was performed. Interruption during startup is tested separately from
persisted chunk resume; a multi-gigabyte interruption across app termination,
two-device AWDL throughput, and Android interoperability were not exercised.
The existing shared transport identity/PIN design and power-loss durability
tradeoffs are unchanged.

The DEBUG launcher executable can retain its UUID across builds; the actual
app code is in `MCPZimChat.debug.dylib`. For this tested build:

- Debug library UUID: `25F5DE56-4232-3ED8-AB85-483D07995D74`.
- Debug library SHA-256: `87e198909a61037d7bd9493090670a0366e45ab163fb12e9b18a70dedd06e565`.
- [Sanitized results](benchmarks/localswarm-update-2026-09-05.json).

Commands:

```sh
swift test --package-path ios/LocalPackages/LocalSwarm
```

Phone builds/install/signature checks use `docs/SIGNED_APP_BUILDS.md`.
Launch the DEBUG probe with `devicectl device process launch` and the
`--environment-variables '{"MCPZIM_SWARM_PROBE":"1"}'` option. Ordinary
launches do not run it. The phone was restored to an ordinary launch afterward and remained alive
throughout the final 45-second health watch. `git diff --check` passed.

Detailed local logs (private device output, not committed):
`/private/tmp/zimfo-localswarm-final-tests.log`,
`/private/tmp/zimfo-localswarm-regression-proof.log`,
`/private/tmp/zimfo-localswarm-model-sharing-final-tests.log`,
`/private/tmp/zimfo-localswarm-phone-verified.log`, and
`/private/tmp/zimfo-localswarm-health-watch.log`.
