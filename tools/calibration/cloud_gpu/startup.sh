#!/bin/bash
# No installs, pulls, downloads, or compilation on paid GPU startup.
set -euo pipefail
umask 077
exec > >(tee -a /var/log/zimfo-gpu.log /dev/console) 2>&1
trap '/sbin/shutdown -h now' EXIT
systemctl disable --now ssh.service ssh.socket 2>/dev/null || true
install -d -m 700 /opt/zimfo-gpu
curl --fail --silent --show-error --retry 3 -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/zimfo-gpu-bootstrap \
  -o /opt/zimfo-gpu/bootstrap.py
curl --fail --silent --show-error --retry 3 -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/zimfo-gpu-config \
  -o /opt/zimfo-gpu/config.json
curl --fail --silent --show-error --retry 3 -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/zimfo-absolute-deadline \
  -o /opt/zimfo-gpu/absolute-deadline.txt
python3 -c 'import hashlib,json; from pathlib import Path; p=Path("/opt/zimfo-gpu"); c=json.loads((p/"config.json").read_text()); assert hashlib.sha256((p/"bootstrap.py").read_bytes()).hexdigest()==c["bootstrap_sha256"]'
# Derive every timer from the same provider absolute cap. Startup consumes
# approved time; it never grants a fresh five hours to the solver.
read -r guest_seconds bootstrap_seconds < <(python3 - <<'PYLIMIT'
import json,time
from datetime import datetime
from pathlib import Path
root=Path('/opt/zimfo-gpu')
config=json.loads((root/'config.json').read_text())
deadline=(root/'absolute-deadline.txt').read_text().strip()
assert deadline == config['absolute_deadline']
hard=datetime.fromisoformat(deadline).timestamp()
remaining=int(hard-time.time())
if remaining < 1800: raise SystemExit('Insufficient runtime remains')
print(remaining-180, remaining-360)
PYLIMIT
)
systemd-run --unit=zimfo-gpu-deadline --on-active="${guest_seconds}s" /sbin/shutdown -h now
if python3 -c 'import json;from pathlib import Path;raise SystemExit(0 if json.loads(Path("/opt/zimfo-gpu/config.json").read_text()).get("mode")=="continuation" else 1)'; then
  curl --fail --silent --show-error --retry 3 -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/zimfo-continuation-controller \
    -o /opt/zimfo-gpu/continuation.py
fi
if python3 -c 'import json;from pathlib import Path;raise SystemExit(0 if json.loads(Path("/opt/zimfo-gpu/config.json").read_text()).get("continuation",{}).get("checkpoint_mode")=="local-spool" else 1)'; then
  curl --fail --silent --show-error --retry 3 -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/zimfo-checkpoint-publisher \
    -o /opt/zimfo-gpu/checkpoint_bridge.py
fi
timeout --signal=TERM --kill-after=120s "${bootstrap_seconds}s" python3 /opt/zimfo-gpu/bootstrap.py
