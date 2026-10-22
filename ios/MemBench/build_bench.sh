#!/usr/bin/env bash
# Run on the Mac. Resolves packages, builds MemBench for the iPhone, installs and launches it.
# Needs the login keychain unlocked in this shell:  security unlock-keychain ~/Library/Keychains/login.keychain-db
set -euo pipefail
ROOT=/Volumes/DevSSD/voice-chat-ios
DEVICE_ID="${DEVICE_ID:-437D2454-02D7-538A-8308-DB7E18D86161}"
cd "$ROOT/app/MemBench"
xcodebuild -project MemBench.xcodeproj -scheme MemBench -configuration Release -sdk iphoneos \
  -derivedDataPath "$ROOT/DerivedData/membench" -clonedSourcePackagesDirPath "$ROOT/SourcePackages" \
  -skipMacroValidation -skipPackagePluginValidation \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration -quiet build
APP="$ROOT/DerivedData/membench/Build/Products/Release-iphoneos/MemBench.app"
xcrun devicectl device install app --device "$DEVICE_ID" "$APP"
xcrun devicectl device process launch --device "$DEVICE_ID" com.luigi.membench-demo
