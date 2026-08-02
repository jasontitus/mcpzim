#!/usr/bin/env bash
# Archive, distribution-sign, validate, and upload Zimfo to App Store Connect.
#
# Authentication prefers the private per-Mac App Store Connect API-key config
# at ~/.config/zimfo/testflight.env. Explicit ASC_* variables override it;
# Xcode's signed-in Apple ID is only the final fallback.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT="$IOS_DIR/MCPZimChat.xcodeproj"
SCHEME="MCPZimChat"
TEAM_ID="A6G8H8NGAM"
BUNDLE_ID="com.tiltastech.zimfo"
MARKETING_VERSION="${MCPZIM_MARKETING_VERSION:-1.0}"
BUILD_NUMBER="${MCPZIM_BUILD_NUMBER:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_ROOT="${MCPZIM_TESTFLIGHT_OUTPUT_DIR:-$IOS_DIR/build-testflight}"
EXISTING_ARCHIVE="${MCPZIM_EXISTING_ARCHIVE:-}"
EXPORT_OPTIONS="$IOS_DIR/ExportOptions-TestFlight.plist"
# Keep package checkouts and downloaded binary artifacts outside the unique
# per-upload DerivedData directory. Besides avoiding repeated Firebase/MLX
# downloads, this preserves SwiftPM's validated mapping for our local llama
# XCFramework. A brand-new SourcePackages directory has intermittently failed
# package resolution with "binary target 'llama' could not be mapped" even
# though the same artifact is valid and builds from the shared cache.
PACKAGE_CACHE="${MCPZIM_PACKAGE_CACHE_DIR:-$IOS_DIR/build-bonsai/SourcePackages}"

# Emergency escape hatch for a broken local CoreSimulator runtime. Xcode's
# asset compiler launches AssetCatalogSimulatorAgent even for a device archive;
# if that runtime is corrupt, actool fails before an otherwise valid archive can
# finish. When explicitly supplied, skip only Assets.xcassets during the build,
# copy the already-compiled, unchanged catalog from a previously accepted app,
# restore the generated icon metadata, and re-sign the completed app before the
# normal signature/export gates. Never use this after changing app icons.
PRECOMPILED_ASSET_APP="${MCPZIM_PRECOMPILED_ASSET_APP:-}"
FALLBACK_SIGNING_IDENTITY="${MCPZIM_FALLBACK_SIGNING_IDENTITY:-Apple Development: Jason Titus (55T347J577)}"
ASSET_BUILD_ARGS=()
if [[ -n "$PRECOMPILED_ASSET_APP" ]]; then
  if [[ ! -d "$PRECOMPILED_ASSET_APP" ]]; then
    echo "testflight: precompiled asset app not found: $PRECOMPILED_ASSET_APP" >&2
    exit 2
  fi
  for asset in Assets.car AppIcon60x60@2x.png AppIcon76x76@2x~ipad.png; do
    if [[ ! -f "$PRECOMPILED_ASSET_APP/$asset" ]]; then
      echo "testflight: precompiled asset missing: $PRECOMPILED_ASSET_APP/$asset" >&2
      exit 2
    fi
  done
  ASSET_BUILD_ARGS+=(
    EXCLUDED_SOURCE_FILE_NAMES=Assets.xcassets
    ASSETCATALOG_COMPILER_APPICON_NAME=
  )
fi

# Prefer a private per-Mac API-key configuration over Xcode's Apple-ID token.
# Xcode credentials have expired mid-session on this machine even after a
# successful upload earlier the same day. The API key is deterministic and
# works for both archive signing and App Store Connect upload. Explicit
# environment variables still win: the config is loaded only when none of the
# three API-key variables were supplied by the caller.
ASC_CONFIG="${MCPZIM_ASC_CONFIG:-$HOME/.config/zimfo/testflight.env}"
AUTH_SOURCE="Xcode account"
if [[ -z "${ASC_KEY_PATH:-}" && -z "${ASC_KEY_ID:-}" \
      && -z "${ASC_ISSUER_ID:-}" && -f "$ASC_CONFIG" ]]; then
  # shellcheck source=/dev/null
  source "$ASC_CONFIG"
  AUTH_SOURCE="private config $ASC_CONFIG"
fi

AUTH_COUNT=0
for variable in ASC_KEY_PATH ASC_KEY_ID ASC_ISSUER_ID; do
  if [[ -n "${!variable:-}" ]]; then
    AUTH_COUNT=$((AUTH_COUNT + 1))
  fi
done

if [[ "$AUTH_COUNT" -ne 0 && "$AUTH_COUNT" -ne 3 ]]; then
  echo "testflight: set ASC_KEY_PATH, ASC_KEY_ID, and ASC_ISSUER_ID together" >&2
  exit 2
fi

if [[ "$AUTH_COUNT" -eq 3 ]]; then
  if [[ ! -f "$ASC_KEY_PATH" ]]; then
    echo "testflight: API key not found: $ASC_KEY_PATH" >&2
    exit 2
  fi
  echo "testflight: authentication = App Store Connect API key $ASC_KEY_ID ($AUTH_SOURCE)"
else
  echo "testflight: authentication = Xcode account"
fi

run_xcodebuild() {
  # Xcode's IPA packaging starts /usr/bin/rsync, which in turn resolves its
  # local server-side helper through PATH. A Homebrew rsync there is not
  # compatible with Apple's -E/--extended-attributes option and makes export
  # fail with "Copy failed". Keep the entire Xcode subprocess on system tools.
  local system_path="/usr/bin:/bin:/usr/sbin:/sbin"
  if [[ "$AUTH_COUNT" -eq 3 ]]; then
    PATH="$system_path" xcodebuild \
      -authenticationKeyPath "$ASC_KEY_PATH" \
      -authenticationKeyID "$ASC_KEY_ID" \
      -authenticationKeyIssuerID "$ASC_ISSUER_ID" \
      "$@"
  else
    PATH="$system_path" xcodebuild "$@"
  fi
}

if [[ -n "$EXISTING_ARCHIVE" ]]; then
  if [[ ! -d "$EXISTING_ARCHIVE" ]]; then
    echo "testflight: existing archive not found: $EXISTING_ARCHIVE" >&2
    exit 2
  fi
  ARCHIVE_PATH="$(cd "$(dirname "$EXISTING_ARCHIVE")" && pwd)/$(basename "$EXISTING_ARCHIVE")"
  RUN_DIR="$(dirname "$ARCHIVE_PATH")"
  APP_INFO="$ARCHIVE_PATH/Products/Applications/MCPZimChat.app/Info.plist"
  MARKETING_VERSION="$(plutil -extract CFBundleShortVersionString raw -o - "$APP_INFO")"
  BUILD_NUMBER="$(plutil -extract CFBundleVersion raw -o - "$APP_INFO")"
  EXPORT_PATH="$RUN_DIR/upload-retry-$(date -u +%Y%m%d%H%M%S)"
  echo "testflight: reusing Zimfo archive $MARKETING_VERSION ($BUILD_NUMBER)"
else
  RUN_DIR="$OUTPUT_ROOT/$MARKETING_VERSION-$BUILD_NUMBER"
  ARCHIVE_PATH="$RUN_DIR/Zimfo.xcarchive"
  EXPORT_PATH="$RUN_DIR/upload"
  DERIVED_DATA_PATH="$RUN_DIR/DerivedData"
  mkdir -p "$RUN_DIR"

  echo "testflight: archiving Zimfo $MARKETING_VERSION ($BUILD_NUMBER)"
  echo "testflight: package cache = $PACKAGE_CACHE"
  run_xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration Release \
    -destination 'generic/platform=iOS' \
    -derivedDataPath "$DERIVED_DATA_PATH" \
    -clonedSourcePackagesDirPath "$PACKAGE_CACHE" \
    -disableAutomaticPackageResolution \
    -archivePath "$ARCHIVE_PATH" \
    -quiet \
    -showBuildTimingSummary \
    -skipMacroValidation \
    -allowProvisioningUpdates \
    DEVELOPMENT_TEAM="$TEAM_ID" \
    CODE_SIGNING_ALLOWED=NO \
    MCPZIM_PACKAGE_CACHE_DIR="$PACKAGE_CACHE" \
    MARKETING_VERSION="$MARKETING_VERSION" \
    CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
    ${ASSET_BUILD_ARGS[@]+"${ASSET_BUILD_ARGS[@]}"} \
    archive
  # CODE_SIGNING_ALLOWED=NO: the archive step must NEVER touch the local
  # keychain — a locked keychain fails headless runs with
  # errSecInternalComponent (real capture 2026-08-02, every framework).
  # All real signing happens at -exportArchive below via the ASC API key +
  # signingStyle=automatic (cloud-managed distribution certs). Same
  # pattern as CastCircle/scripts/ship-testflight.sh, which ships
  # routinely with NO distribution cert in the keychain at all.
fi

if [[ -n "$PRECOMPILED_ASSET_APP" && -z "$EXISTING_ARCHIVE" ]]; then
  APP_PATH="$ARCHIVE_PATH/Products/Applications/MCPZimChat.app"
  APP_INFO="$APP_PATH/Info.plist"
  echo "testflight: restoring unchanged compiled app icons from $PRECOMPILED_ASSET_APP"
  cp "$PRECOMPILED_ASSET_APP/Assets.car" "$APP_PATH/Assets.car"
  cp "$PRECOMPILED_ASSET_APP/AppIcon60x60@2x.png" "$APP_PATH/AppIcon60x60@2x.png"
  cp "$PRECOMPILED_ASSET_APP/AppIcon76x76@2x~ipad.png" "$APP_PATH/AppIcon76x76@2x~ipad.png"

  /usr/libexec/PlistBuddy -c 'Delete :CFBundleIcons' "$APP_INFO" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons dict' "$APP_INFO"
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons:CFBundlePrimaryIcon dict' "$APP_INFO"
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons:CFBundlePrimaryIcon:CFBundleIconFiles array' "$APP_INFO"
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons:CFBundlePrimaryIcon:CFBundleIconFiles:0 string AppIcon60x60' "$APP_INFO"
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons:CFBundlePrimaryIcon:CFBundleIconName string AppIcon' "$APP_INFO"

  /usr/libexec/PlistBuddy -c 'Delete :CFBundleIcons~ipad' "$APP_INFO" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons~ipad dict' "$APP_INFO"
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons~ipad:CFBundlePrimaryIcon dict' "$APP_INFO"
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons~ipad:CFBundlePrimaryIcon:CFBundleIconFiles array' "$APP_INFO"
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons~ipad:CFBundlePrimaryIcon:CFBundleIconFiles:0 string AppIcon60x60' "$APP_INFO"
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons~ipad:CFBundlePrimaryIcon:CFBundleIconFiles:1 string AppIcon76x76' "$APP_INFO"
  /usr/libexec/PlistBuddy -c 'Add :CFBundleIcons~ipad:CFBundlePrimaryIcon:CFBundleIconName string AppIcon' "$APP_INFO"

  /usr/bin/codesign --force --sign "$FALLBACK_SIGNING_IDENTITY" \
    --preserve-metadata=identifier,entitlements,requirements,flags \
    --timestamp=none "$APP_PATH"
  /usr/bin/codesign --verify --deep --strict "$APP_PATH"
  echo "testflight: precompiled asset fallback ready; normal signature gate follows"
fi

# The archive is deliberately UNSIGNED (see CODE_SIGNING_ALLOWED=NO above);
# signing happens at export. Keep the bundle-identity half of the gate, and
# run the full signature gate only when the archive carries a signature
# (i.e. someone reverts to archive-time signing).
ARCHIVED_APP="$ARCHIVE_PATH/Products/Applications/MCPZimChat.app"
if /usr/bin/codesign -dv "$ARCHIVED_APP" >/dev/null 2>&1; then
  "$SCRIPT_DIR/verify-app-signature.sh" "$ARCHIVED_APP" "$BUNDLE_ID" "$TEAM_ID"
else
  ACTUAL_BUNDLE="$(plutil -extract CFBundleIdentifier raw -o - "$ARCHIVED_APP/Info.plist")"
  if [[ "$ACTUAL_BUNDLE" != "$BUNDLE_ID" ]]; then
    echo "testflight: bundle ID is $ACTUAL_BUNDLE; expected $BUNDLE_ID" >&2
    exit 1
  fi
  echo "testflight: archive unsigned by design — distribution signing + validation happen at export/upload"
fi

# Prebuilt binary frameworks (Firebase/GoogleAppMeasurement…) come out of an
# unsigned archive with AD-HOC linker signatures. -exportArchive treats
# anything signed as already handled and skips them — and App Store
# validation rejects ad-hoc ("Invalid Signature … not properly signed",
# real capture 2026-08-02). Stripping needs NO identity/keychain; export
# then re-signs every framework with the distribution cert like the app.
FRAMEWORKS_DIR="$ARCHIVED_APP/Frameworks"
if [[ -d "$FRAMEWORKS_DIR" ]]; then
  while IFS= read -r -d '' fw; do
    /usr/bin/codesign --remove-signature "$fw" 2>/dev/null || true
  done < <(find "$FRAMEWORKS_DIR" -maxdepth 1 \( -name "*.framework" -o -name "*.dylib" \) -print0)
  echo "testflight: stripped ad-hoc signatures from embedded frameworks for export re-sign"
fi

echo "testflight: uploading Zimfo $MARKETING_VERSION ($BUILD_NUMBER)"
run_xcodebuild \
  -exportArchive \
  -archivePath "$ARCHIVE_PATH" \
  -exportPath "$EXPORT_PATH" \
  -exportOptionsPlist "$EXPORT_OPTIONS" \
  -allowProvisioningUpdates

echo "testflight: upload submitted · Zimfo $MARKETING_VERSION ($BUILD_NUMBER)"
echo "testflight: archive retained at $ARCHIVE_PATH"
