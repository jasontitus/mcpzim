# Vendored LocalSwarmEngine

Vendored from https://github.com/jasontitus/localswarm at commit
`b43f007` on branch `claude/mcpzim-local-swarm-fex4ts` (engine `Sources/` +
`Tests/` + `Package.swift` only — the upstream repo's own SwiftUI app,
Android app, and Go peer are not vendored). That commit adds directory
shares (`ShareItem` relative paths + `hostFiles` expansion), which Zimfo
uses to hand over multi-file voice-model folders.

LocalSwarmEngine is the local-first peer-to-peer file swarming engine Zimfo
uses for **Nearby Sharing**: seeding your ZIM library to nearby devices over
AWDL / peer-to-peer Wi-Fi (TCP+TLS and QUIC transports), and pulling a
friend's ZIMs from every nearby source in parallel with SHA-256 per-chunk
verification and resume.

Local path (not a git URL) for the same reasons as the other vendored
packages here: hermetic builds with no auth against a private repo, and the
app pin moves only when this directory is deliberately re-synced.

## Re-syncing with upstream

```sh
# from the mcpzim repo root, with a localswarm checkout beside it:
rm -rf ios/LocalPackages/LocalSwarm/Sources ios/LocalPackages/LocalSwarm/Tests
cp -r ../localswarm/Sources ../localswarm/Tests ios/LocalPackages/LocalSwarm/
cp ../localswarm/Package.swift ../localswarm/LICENSE ../localswarm/NOTICE \
   ios/LocalPackages/LocalSwarm/
# then update the pinned commit hash at the top of this file
```

No local patches are applied today. If a patch ever becomes necessary,
document it here the way `LocalPackages/FluidAudio/MCPZIM_PATCHES.md` does,
and prefer upstreaming it to jasontitus/localswarm first.

## License

Apache-2.0 (see `LICENSE` / `NOTICE` in this directory). The rest of the
mcpzim repo is MIT; this directory keeps its upstream license.
