import Foundation

/// Display-side Simplified → Traditional conversion.
///
/// The model and the TTS work in Simplified on purpose; only what the user reads is
/// converted. ICU's Hans-Hant transform is character level, so Taiwan-specific vocabulary
/// (软件 → 軟體 rather than 軟件) is not rewritten.
enum ZhTW {
    private static let transform = StringTransform("Hans-Hant")

    static func convert(_ text: String) -> String {
        text.applyingTransform(transform, reverse: false) ?? text
    }
}
