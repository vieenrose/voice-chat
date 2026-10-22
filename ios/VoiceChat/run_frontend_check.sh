#!/usr/bin/env bash
# Run on the Mac: compiles the TTS text frontend with a host swiftc and compares it with golden.json.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${TMPDIR:-/tmp}/FrontendCheck"
swiftc -O -o "$OUT" VoiceChat/Local/TTS/{NumberWords,TextNorm,G2PW,G2PEn,PrimeFrontend}.swift checks/FrontendCheck/main.swift
"$OUT" checks/FrontendCheck/golden.json VoiceChat/Resources/tts
