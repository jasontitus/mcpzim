#!/usr/bin/env bash
# Fail closed before a Zimfo app is launched or installed. Privacy permission
# testing is invalid when an artifact is unsigned or ad-hoc signed.

set -euo pipefail

APP="${1:?usage: verify-app-signature.sh <app> <expected-bundle-id> [expected-team-id]}"
EXPECTED_BUNDLE="${2:?usage: verify-app-signature.sh <app> <expected-bundle-id> [expected-team-id]}"
EXPECTED_TEAM="${3:-A6G8H8NGAM}"

if [[ ! -d "$APP" ]]; then
  echo "signature gate: app not found: $APP" >&2
  exit 1
fi

INFO_PLIST="$APP/Info.plist"
if [[ -f "$APP/Contents/Info.plist" ]]; then
  INFO_PLIST="$APP/Contents/Info.plist"
fi

ACTUAL_BUNDLE="$(plutil -extract CFBundleIdentifier raw -o - "$INFO_PLIST")"
DETAILS="$(codesign -dvvv "$APP" 2>&1)"

if [[ "$ACTUAL_BUNDLE" != "$EXPECTED_BUNDLE" ]]; then
  echo "signature gate: bundle ID is $ACTUAL_BUNDLE; expected $EXPECTED_BUNDLE" >&2
  exit 1
fi
if grep -Eq 'Signature=adhoc|flags=.*(adhoc|linker-signed)' <<<"$DETAILS"; then
  echo "signature gate: refusing ad-hoc/linker-signed app" >&2
  echo "$DETAILS" | grep -E 'Identifier=|flags=|Signature=|TeamIdentifier=' >&2
  exit 1
fi
if ! grep -q "Identifier=$EXPECTED_BUNDLE" <<<"$DETAILS"; then
  echo "signature gate: code-signing identifier does not match $EXPECTED_BUNDLE" >&2
  echo "$DETAILS" | grep -E 'Identifier=|flags=|Signature=|TeamIdentifier=' >&2
  exit 1
fi
if ! grep -q "TeamIdentifier=$EXPECTED_TEAM" <<<"$DETAILS"; then
  echo "signature gate: TeamIdentifier is not $EXPECTED_TEAM" >&2
  echo "$DETAILS" | grep -E 'Identifier=|flags=|Signature=|TeamIdentifier=' >&2
  exit 1
fi

# macOS can validate the complete local trust chain. An iOS development app's
# trust/provisioning decision is made by the target device during install; a
# host-side strict verify may report CSSMERR_TP_NOT_TRUSTED even though its CMS
# signature and provisioning profile are present. The checks above still reject
# unsigned/ad-hoc iOS artifacts, and devicectl is the final provisioning gate.
if [[ -d "$APP/Contents" ]]; then
  codesign --verify --deep --strict --verbose=2 "$APP"
fi
echo "signature gate: OK · $EXPECTED_BUNDLE · team $EXPECTED_TEAM"
