#!/usr/bin/env bash
# Run on the Mac. Writes the LAN URL (with token) into the Mac's copy of LocalConfig.swift,
# builds for the connected iPhone, installs and launches it.
# Needs the login keychain unlocked in this shell:  security unlock-keychain ~/Library/Keychains/login.keychain-db
set -euo pipefail
ROOT=/Volumes/DevSSD/voice-chat-ios
DEVICE_UDID="${DEVICE_UDID:-00008120-000148CA2EEB401E}"
DEVICE_ID="${DEVICE_ID:-437D2454-02D7-538A-8308-DB7E18D86161}"
LAN_IP="${LAN_IP:-$(ipconfig getifaddr en0)}"
TOKEN="$(cat "$ROOT/server/lan_token")"
cd "$ROOT/app/VoiceChat"
printf 'enum LocalConfig {\n    static let serverURL = "ws://%s:8766/v1/realtime?token=%s"\n}\n' "$LAN_IP" "$TOKEN" > VoiceChat/LocalConfig.swift
xcodebuild -project VoiceChat.xcodeproj -scheme VoiceChat -configuration Release -sdk iphoneos \
  -derivedDataPath "$ROOT/DerivedData/voicechat" -clonedSourcePackagesDirPath "$ROOT/SourcePackages" \
  -skipMacroValidation -skipPackagePluginValidation \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration -quiet build
APP="$ROOT/DerivedData/voicechat/Build/Products/Release-iphoneos/VoiceChat.app"
xcrun devicectl device install app --device "$DEVICE_ID" "$APP"
# The X-ASR bundle is not downloadable in-app: push it once into the app container.
if [[ "${PUSH_ASR:-0}" == 1 ]]; then
  xcrun devicectl device copy to --device "$DEVICE_ID" --domain-type appDataContainer \
    --domain-identifier com.luigi.voicechat-demo --source "$ROOT/models/xasr160" --destination Library/Caches/xasr
fi
xcrun devicectl device process launch --device "$DEVICE_ID" com.luigi.voicechat-demo
