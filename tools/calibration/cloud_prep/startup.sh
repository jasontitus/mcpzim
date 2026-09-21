#!/bin/bash
# CPU-only preparation. The API supplies a second, independent one-hour limit.
set -euo pipefail
umask 077
trap '/sbin/shutdown -h now || true' EXIT
exec > >(tee -a /var/log/zimfo-prep.log /dev/console) 2>&1
systemd-run --unit=zimfo-prep-deadline --on-active=55m /sbin/shutdown -h now
systemctl disable --now ssh.service ssh.socket 2>/dev/null || true
install -d -m 700 /opt/zimfo-prep
curl --fail --silent --show-error --retry 3 \
  -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/zimfo-prep-bootstrap \
  -o /opt/zimfo-prep/bootstrap.py
python3 - <<'PY'
import hashlib, json, urllib.request
request = urllib.request.Request(
    'http://metadata.google.internal/computeMetadata/v1/instance/attributes/zimfo-prep-config',
    headers={'Metadata-Flavor': 'Google'})
with urllib.request.urlopen(request, timeout=20) as response:
    config = json.loads(response.read(1024 * 1024))
with open('/opt/zimfo-prep/bootstrap.py', 'rb') as source:
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(chunk)
    actual = digest.hexdigest()
if actual != config['bootstrap_sha256']:
    raise RuntimeError('Bootstrap code differs from reviewed config')
PY
timeout --signal=TERM --kill-after=30s 3000s python3 /opt/zimfo-prep/bootstrap.py
