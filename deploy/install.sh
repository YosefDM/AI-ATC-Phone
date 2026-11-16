#!/usr/bin/env bash
# Install or update the AI ATC phone line on an Ubuntu droplet that already runs Caddy.
# Run from the repo checkout:  sudo bash deploy/install.sh
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST=/opt/ai-atc-phone

apt-get update -qq
apt-get install -y -qq python3-venv python3-dev rsync curl >/dev/null

id aiatc >/dev/null 2>&1 || useradd --system --home "$DEST" --shell /usr/sbin/nologin aiatc
mkdir -p "$DEST"
rsync -a --delete --exclude /.git --exclude /.venv --exclude /.env --exclude /data --exclude /voices --exclude /logs --exclude __pycache__ \
  "$SRC/" "$DEST/"
mkdir -p "$DEST/data" "$DEST/voices" "$DEST/logs"

python3 -m venv "$DEST/.venv"
"$DEST/.venv/bin/pip" install -q --upgrade pip
"$DEST/.venv/bin/pip" install -q -r "$DEST/requirements.txt"
bash "$DEST/scripts/fetch_data.sh"

if [ ! -f "$DEST/.env" ]; then
  cp "$DEST/.env.example" "$DEST/.env"
  sed -i "s/^APP_SECRET=.*/APP_SECRET=$(openssl rand -hex 32)/" "$DEST/.env"
  echo ">>> Created $DEST/.env: fill in OPENAI_API_KEY, TWILIO_* and PUBLIC_BASE_URL, then: systemctl restart ai-atc"
fi
chmod 600 "$DEST/.env"
chown -R aiatc:aiatc "$DEST"

# swap helps a 1 GB droplet absorb the voice model load
if [ "$(swapon --show | wc -l)" -eq 0 ] && [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi

cp "$DEST/deploy/ai-atc.service" /etc/systemd/system/ai-atc.service
systemctl daemon-reload
systemctl enable ai-atc >/dev/null
systemctl restart ai-atc
sleep 25
curl -fsS http://127.0.0.1:8090/health && echo " <- ai-atc is up"
