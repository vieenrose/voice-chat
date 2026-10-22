#!/usr/bin/env bash
# Compiles the interruption logic with a fake LLM/TTS/speaker and runs the checks (needs swiftc, i.e. a Mac).
set -euo pipefail
cd "$(dirname "$0")"
OUT="${TMPDIR:-/tmp}/bargein_check"
swiftc -swift-version 5 -O VoiceChat/Local/*.swift checks/BargeInCheck/main.swift -o "$OUT"
"$OUT"
