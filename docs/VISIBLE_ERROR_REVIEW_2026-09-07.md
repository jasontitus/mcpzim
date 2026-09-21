# Visible conversation errors: logging and copying

The user reasonably expected the debug log to contain an error they saw.
The affected log has no explicit voice error, and its missing wording cannot
be reconstructed. The UI previously offered no copy action on its alerts.

## Changes

- Capture chat errors, library errors, model-load failures and voice error
  states centrally when assigned, tagged `UIError` with the exact message
  and its source. This uses `ChatSession.debug` and the existing archive/
  opt-in upload pipeline, even if the view disappears before it renders.
- Log displayed voice fallback notices separately as notices, not failures.
- Add “Copy error” to chat/library alerts and an accessible copy button in
  the compact voice row. Copy includes the full message and newlines.
- Allow text selection in the voice preview.

## Adversarial self-review

- Do not depend on SwiftUI `onAppear`/`onChange` for logging: a transient
  state can disappear before a render. Property observers capture it first.
- Do not read the optional alert binding inside its copy action: dismissal
  can clear it. SwiftUI's `presenting` payload captures the error being shown.
- Do not put the voice copy action in an extra vertical row: the default
  sheet is only 72 points high. Keep the button beside the message and copy
  full text despite the two-line preview.
- Suppress reassignment of the same outstanding error, but log the same
  message again after dismissal or recovery. Keep multiline messages intact.
- No clipboard writes occur until the user presses Copy. No new remote
  destination, upload preference or log-retention policy is introduced.
- Scope is app-controlled conversation/library/model error states and voice
  notices. This does not capture arbitrary OS-owned alerts or retroactively
  recover an unlogged message. Existing asynchronous archive durability still
  applies; an immediate process kill can precede queued writes.

`VisibleErrorLoggingTests` exercises dismissal, duplicate assignments,
recurrence, multiline messages and library/model source labels without
loading a model. The UI copy payload is reviewed separately from the log
test; no synthetic failure was injected into the user's phone conversation.

Validation: the focused Mac test passed, the signed iOS Debug build passed,
and the app/framework signature gate passed. Evidence is retained in
`/private/tmp/visible-errors-mac-tests.log`,
`/private/tmp/visible-errors-ios-final.log`, and
`/private/tmp/visible-errors-ios-signature.log`.

Installed on the phone; the 45-second process check passed. A new system
JetsamEvent appeared during verification. Inspection shows Zimfo listed as
suspended, with no termination reason or kill delta; the report records other
process reclamation. Do not mistake `largestProcess: MCPZimChat` alone for an
app crash, or the successful startup check for an absence of system memory
pressure. Evidence: `/private/tmp/visible-errors-phone-deploy.log`.
