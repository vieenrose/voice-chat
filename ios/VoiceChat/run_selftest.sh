#!/usr/bin/env bash
# Run on the Mac with the iPhone unlocked and plugged in. Launches the app in -selftest mode
# (a wav replaces the microphone) and prints the pipeline timings from the console.
set -euo pipefail
DEVICE_ID="${DEVICE_ID:-437D2454-02D7-538A-8308-DB7E18D86161}"
OUT=/Volumes/DevSSD/voice-chat-ios/selftest.out
xcrun devicectl device process launch --terminate-existing --console --device "$DEVICE_ID" com.luigi.voicechat-demo -- -selftest > "$OUT" 2>&1 &
PID=$!
sleep "${SELFTEST_SECS:-150}"
kill $PID 2>/dev/null || true
grep '\[local\]' "$OUT" || tail -20 "$OUT"
