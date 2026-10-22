import Foundation
import MLXLMCommon

/// Where the local models live and how they get there. Everything except the X-ASR files is
/// downloaded on first use; the X-ASR bundle is pushed with `devicectl` (see README).
enum LocalModels {
    static let ttsURL = URL(string: "https://huggingface.co/Luigi/PrimeTTS/resolve/main/v21_mbistft_16k/primetts_v21_3voice.onnx")!
    static let g2pwURL = URL(string: "https://huggingface.co/Luigi/PrimeTTS/resolve/main/g2pw_int8/g2pw_int8.onnx")!
    static let llmRepo = "prism-ml/Ternary-Bonsai-4B-mlx-2bit"

    static var caches: URL { FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0] }

    static func asrFile(_ rel: String) throws -> URL {
        let url = caches.appendingPathComponent("xasr/" + rel)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LocalError.missing("語音辨識模型檔：\(rel)")
        }
        return url
    }

    /// Downloads once into Caches; returns the local path.
    static func fetch(_ url: URL) async throws -> String {
        let dest = caches.appendingPathComponent(url.lastPathComponent)
        if !FileManager.default.fileExists(atPath: dest.path) {
            let (tmp, _) = try await URLSession.shared.download(from: url)
            try FileManager.default.moveItem(at: tmp, to: dest)
        }
        return dest.path
    }

    static func resource(_ name: String) throws -> URL {
        // The synchronized Xcode group copies resources flat into the bundle root.
        guard let url = Bundle.main.url(forResource: name, withExtension: nil)
                ?? Bundle.main.resourceURL?.appendingPathComponent("tts/" + name),
              FileManager.default.fileExists(atPath: url.path) else {
            throw LocalError.missing("資源 \(name)")
        }
        return url
    }
}

enum LocalError: Error, LocalizedError {
    case missing(String)
    var errorDescription: String? {
        switch self { case .missing(let what): return "缺少\(what)" }
    }
}
