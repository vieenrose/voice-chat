// Placeholder. build_app.sh overwrites the copy on the Mac with the real LAN URL (it carries
// the secret token), so never commit a version of this file that contains a token.
enum LocalConfig {
    static let serverURL = "ws://192.168.0.36:8766/v1/realtime?token="
}
