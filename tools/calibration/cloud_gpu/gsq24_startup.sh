#!/bin/bash
# CPU prepared every dependency and byte; never install or pull on the GPU.
set -euo pipefail
umask 077
exec > >(tee -a /var/log/zimfo-gsq24.log /dev/console) 2>&1
trap '/sbin/shutdown -h now' EXIT
systemctl disable --now ssh.service ssh.socket 2>/dev/null || true
install -d -m 700 /opt/zimfo-gsq24
for pair in config.json:zimfo-gsq24-config gsq24_guest.py:zimfo-gsq24-guest gpu_common.py:zimfo-gsq24-common; do
  filename=${pair%%:*}; attribute=${pair#*:}
  curl --fail --silent --show-error --retry 3 --max-time 30 -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${attribute}" -o "/opt/zimfo-gsq24/${filename}"
done
read -r guest_seconds supervisor_seconds < <(python3 - <<'PY'
import hashlib,json,time
from pathlib import Path
from datetime import datetime
root=Path('/opt/zimfo-gsq24');config=json.loads((root/'config.json').read_text())
body={k:v for k,v in config.items() if k!='config_sha256'}
assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==config['config_sha256']
for name in ('gsq24_guest.py','gpu_common.py'):assert hashlib.sha256((root/name).read_bytes()).hexdigest()==config['files'][name]
remaining=int(datetime.fromisoformat(config['absolute_deadline']).timestamp()-time.time())
assert remaining>=900
print(remaining-180,remaining-240)
PY
)
systemd-run --unit=zimfo-gsq24-deadline --on-active="${guest_seconds}s" /sbin/shutdown -h now
timeout --signal=TERM --kill-after=55s "${supervisor_seconds}s" python3 /opt/zimfo-gsq24/gsq24_guest.py
