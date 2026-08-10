# Repository operational notes

## Zimfo iOS TestFlight

Do not reconstruct or improvise the TestFlight workflow from chat history.
Before building or uploading, read `docs/SIGNED_APP_BUILDS.md` and use the
repository script:

```sh
ios/scripts/testflight-upload.sh
```

On the Mac Studio, that script automatically loads the working App Store
Connect API-key mapping from the private, uncommitted file
`~/.config/zimfo/testflight.env`. Keep bundle ID `com.tiltastech.zimfo` and
team `A6G8H8NGAM`. Never commit the private key or local authentication file.

If archiving succeeds but export or upload fails, do not compile again. Retry
the retained archive with `MCPZIM_EXISTING_ARCHIVE` as documented in the
runbook. Treat the upload as accepted only after the signature gate, Xcode's
`Upload succeeded` and `EXPORT SUCCEEDED` messages, and the script's final
`upload submitted` message are present.
