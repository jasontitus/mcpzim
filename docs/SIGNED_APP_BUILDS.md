# Signed Zimfo builds

This is the canonical build and deployment path for the iOS and macOS apps.
Both apps use privacy-protected services (microphone, speech recognition, and
location), so signing is part of correctness, not just distribution.

## Non-negotiable rule

Never launch or deploy a runnable app built with `CODE_SIGNING_ALLOWED=NO`,
`CODE_SIGN_IDENTITY=-`, or a linker/ad-hoc signature. An ad-hoc signature's
designated requirement changes with the executable, so macOS TCC can forget a
microphone grant after every rebuild and repeatedly show the permission sheet.

Compile-only experiments may disable signing only when their product will not
be launched. Before launching either app, run the signature gate below.

The project currently uses:

- Team ID: `A6G8H8NGAM`
- iOS bundle ID: `com.tiltastech.zimfo`
- macOS bundle ID: `org.mcpzim.MCPZimChatMac`
- macOS signing identity: `Developer ID Application: Jason Titus (A6G8H8NGAM)`

The Mac target's team is pinned in both `ios/project.yml` and the generated
Xcode project. If `xcodegen generate` is run, verify that the target still has
this team before building.

## macOS: build, verify, then launch

From the repository root:

```sh
pkill -x MCPZimChatMac 2>/dev/null || true

xcodebuild \
  -project ios/MCPZimChat.xcodeproj \
  -scheme MCPZimChatMac \
  -configuration Release \
  -destination 'platform=macOS' \
  -derivedDataPath ios/build-mac-bonsai \
  ARCHS=arm64 \
  ONLY_ACTIVE_ARCH=YES \
  DEVELOPMENT_TEAM=A6G8H8NGAM \
  CODE_SIGN_STYLE=Manual \
  'CODE_SIGN_IDENTITY=Developer ID Application: Jason Titus (A6G8H8NGAM)' \
  OTHER_CODE_SIGN_FLAGS=--timestamp=none \
  build

ios/scripts/verify-app-signature.sh \
  ios/build-mac-bonsai/Build/Products/Release/MCPZimChatMac.app \
  org.mcpzim.MCPZimChatMac

open ios/build-mac-bonsai/Build/Products/Release/MCPZimChatMac.app
```

Do not run `open` if the verification script fails. In particular, these are
failures:

- `Signature=adhoc`
- `flags` contains `adhoc` or `linker-signed`
- `TeamIdentifier=not set`
- `Identifier=MCPZimChatMac` instead of the full bundle ID

The valid result has `Identifier=org.mcpzim.MCPZimChatMac` and
`TeamIdentifier=A6G8H8NGAM`.

## iPhone: build, verify, install, and watch

Automatic signing supplies the development provisioning profile for the
connected iPhone:

```sh
xcodebuild \
  -project ios/MCPZimChat.xcodeproj \
  -scheme MCPZimChat \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath ios/build-bonsai \
  DEVELOPMENT_TEAM=A6G8H8NGAM \
  CODE_SIGN_STYLE=Automatic \
  -allowProvisioningUpdates \
  build

ios/scripts/verify-app-signature.sh \
  ios/build-bonsai/Build/Products/Debug-iphoneos/MCPZimChat.app \
  com.tiltastech.zimfo

MCPZIM_APP_PATH="$PWD/ios/build-bonsai/Build/Products/Debug-iphoneos/MCPZimChat.app" \
  ios/scripts/mcp-deploy-verify.sh
```

`mcp-deploy-verify.sh` installs and launches through `devicectl`, then checks
that the process stays alive. Its default device UUID is the current iPhone;
override it with `MCPZIM_DEVICE_UUID` when needed.

## TestFlight: archive, sign, and upload

The TestFlight build uses the same iOS target and bundle ID as phone
development. Do not create a second target or change the bundle ID just for
distribution.

Last verified on the Mac Studio on 2026-07-16: Zimfo 1.0 build
`20260717021449` was accepted by App Store Connect using this workflow.

One-time App Store Connect setup:

- An iOS app record must exist for bundle ID `com.tiltastech.zimfo`.
- Prefer an App Store Connect API key. Xcode's saved Apple-ID token has gone
  stale between archive and upload on this Mac even after working earlier the
  same day.
- Agreements, tax, or banking notices in App Store Connect must not be
  blocking uploads.

### Authentication that works reliably on the Mac Studio

`testflight-upload.sh` automatically loads this private per-Mac file when the
three `ASC_*` variables are not already present:

```text
~/.config/zimfo/testflight.env
```

The file must be mode `600` and contain:

```sh
ASC_KEY_PATH="$HOME/.appstoreconnect/private_keys/AuthKey_YOUR_KEY_ID.p8"
ASC_KEY_ID="YOUR_KEY_ID"
ASC_ISSUER_ID="YOUR_ISSUER_UUID"
```

The `.p8` private key and this local configuration must never be committed.
The environment variables remain supported and override the local file. Set
`MCPZIM_ASC_CONFIG` only when using a different private configuration path.

From the repository root:

```sh
ios/scripts/testflight-upload.sh
```

The script:

1. Creates a Release archive for a generic iOS device, reusing the validated
   package checkout and binary-artifact cache under
   `ios/build-bonsai/SourcePackages`. Override that location with
   `MCPZIM_PACKAGE_CACHE_DIR` only when deliberately using another complete
   Xcode SourcePackages cache.
2. Uses automatic signing for team `A6G8H8NGAM`; Xcode may create a
   cloud-managed Apple Distribution certificate and App Store profile.
3. Runs the same signature gate used by development builds.
4. Exports with `method=app-store-connect` and `destination=upload`, which
   validates and submits the build to App Store Connect.
5. Waits for the exact uploaded build number to become eligible for internal
   testing, assigns it to the existing `InternalTesters` group, and verifies
   that relationship through the App Store Connect API. Set
   `MCPZIM_SKIP_INTERNAL_ASSIGNMENT=1` only when deliberately uploading a
   build that should not be distributed internally.

The script intentionally gives Xcode a system-only `PATH`. During IPA export,
Apple's `/usr/bin/rsync` starts a helper through `PATH`; resolving that helper
to Homebrew rsync 3.x instead causes `--extended-attributes: unknown option`
and Xcode reports only `exportArchive Copy failed`.

By default, the marketing version is `1.0` and the build number is the UTC
timestamp, so repeat uploads do not reuse a build number. Override either when
needed:

```sh
MCPZIM_MARKETING_VERSION=1.1 \
MCPZIM_BUILD_NUMBER=42 \
  ios/scripts/testflight-upload.sh
```

For a one-off upload on a machine without the private config, set all three
variables:

```sh
ASC_KEY_PATH="$HOME/.appstoreconnect/private_keys/AuthKey_EXAMPLE.p8" \
ASC_KEY_ID=EXAMPLE \
ASC_ISSUER_ID=00000000-0000-0000-0000-000000000000 \
  ios/scripts/testflight-upload.sh
```

Never commit the `.p8` key or its issuer metadata. The archive and upload logs
remain under `ios/build-testflight/<version>-<build>/`.

If the archive succeeded but an account or network problem blocked the upload,
reuse it instead of rebuilding:

```sh
MCPZIM_EXISTING_ARCHIVE="$PWD/ios/build-testflight/1.0-<build>/Zimfo.xcarchive" \
  ios/scripts/testflight-upload.sh
```

This retry uses the same private config automatically. Do not rebuild merely
because export reports `Failed to Use Accounts`: the signed archive is valid,
and retrying it with the API-key config avoids another multi-minute compile.

Successful completion must include all of these lines:

```text
signature gate: OK · com.tiltastech.zimfo · team A6G8H8NGAM
Progress 98%: Upload succeeded.
** EXPORT SUCCEEDED **
testflight: upload submitted · Zimfo <version> (<build>)
```

After that, App Store Connect processes the build asynchronously. Missing
dSYM warnings for precompiled Firebase/Google frameworks are nonblocking; an
`Upload succeeded` result means the TestFlight package was accepted.

### Emergency asset-catalog fallback

If archive compilation succeeds but `actool` fails because
`AssetCatalogSimulatorAgent` cannot load Apple libraries from an installed
CoreSimulator runtime, first reboot the Mac and verify or reinstall that
runtime. When a reboot is temporarily impossible and app icons have not
changed since a previously accepted archive, the upload script has an explicit
narrow fallback:

```sh
MCPZIM_BUILD_NUMBER=<retained-build-number> \
MCPZIM_PRECOMPILED_ASSET_APP="$PWD/ios/build-testflight/1.0-<accepted-build>/Zimfo.xcarchive/Products/Applications/MCPZimChat.app" \
  ios/scripts/testflight-upload.sh
```

This excludes only `Assets.xcassets`, restores `Assets.car` and the generated
icon files and plist metadata from that accepted app, re-signs the archive,
and then runs the same signature, export, and upload gates. Do not use it after
an icon or asset-catalog change. This is a recovery path, not a substitute for
repairing the corrupt simulator runtime.

The upload script also passes its shared package-cache directory into Xcode as
`MCPZIM_PACKAGE_CACHE_DIR`. The Crashlytics dSYM phase uses that setting when
locating Firebase's `Crashlytics/run` helper, while ordinary Xcode builds still
fall back to the package directory beneath their own DerivedData root.

## Privacy-permission recovery

Keep the bundle ID and signing team stable. A normal rebuild should reuse the
existing privacy grants and should not prompt again.

If a bad ad-hoc build was launched and macOS TCC is already confused:

1. Quit every copy of the app.
2. Build and pass the signed-app verification gate.
3. Reset only this app's stale microphone record once:

   ```sh
   tccutil reset Microphone org.mcpzim.MCPZimChatMac
   ```

4. Launch the verified signed app and click **Allow** once.

Do not repeatedly reset TCC during ordinary development; that guarantees
another prompt and hides whether signing persistence is working.

## Preflight when signing fails

List available identities:

```sh
security find-identity -v -p codesigning
```

Use an unexpired identity whose team is `A6G8H8NGAM`. As of July 15, 2026,
the valid Mac identity is the Developer ID identity shown above. If it is
replaced, update this document and the Mac build command together.
