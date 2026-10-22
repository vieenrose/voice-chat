#!/usr/bin/env bash
# Expose the local Realtime server to phones on the same Wi-Fi, behind a secret token.
# The token lives in server/lan_token (created on first run, mode 600, never committed).
# The app gets it at build time: see ios/VoiceChat/README or build_app.sh.
set -euo pipefail

ROOT=/Volumes/DevSSD/voice-chat-ios
cd "$ROOT/server"
if [[ ! -s lan_token ]]; then
  (umask 077 && "$ROOT/venv-s2s1/bin/python" -c "import secrets; print(secrets.token_urlsafe(18))" > lan_token)
fi
LAN_IP="${LAN_IP:-$(ipconfig getifaddr en0)}"
export S2S_LAN_TOKEN_FILE="$ROOT/server/lan_token"
exec "$ROOT/venv-s2s1/bin/python" "$ROOT/server/lan_proxy.py" "$LAN_IP" "${LAN_PORT:-8766}"
