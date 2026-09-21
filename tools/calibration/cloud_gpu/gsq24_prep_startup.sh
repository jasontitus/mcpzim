#!/bin/bash
set -euo pipefail
umask 077
exec > >(tee -a /var/log/zimfo-gsq24-prep.log /dev/console) 2>&1
trap '/sbin/shutdown -h now' EXIT
systemd-run --unit=zimfo-gsq24-prep-deadline --on-active=55m /sbin/shutdown -h now
systemctl disable --now ssh.service ssh.socket 2>/dev/null || true
install -d -m 700 /opt/zimfo-gsq24-prep
for pair in config.json:zimfo-gsq24-prep-config gsq24_prep_guest.py:zimfo-gsq24-prep-guest cpu_common.py:zimfo-gsq24-cpu-common; do
  filename=${pair%%:*}; attribute=${pair#*:}
  curl --fail --silent --show-error --retry 3 --max-time 30 -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${attribute}" \
    -o "/opt/zimfo-gsq24-prep/${filename}"
done
python3 - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('/opt/zimfo-gsq24-prep');config=json.loads((root/'config.json').read_text())
for name in ('gsq24_prep_guest.py','cpu_common.py'):
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==config['files'][name]
PY
timeout --signal=TERM --kill-after=60s 3150s python3 /opt/zimfo-gsq24-prep/gsq24_prep_guest.py
