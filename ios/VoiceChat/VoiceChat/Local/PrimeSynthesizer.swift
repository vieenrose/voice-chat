import Foundation

/// PrimeTTS v2.1 (VITS, 3 voices, 16 kHz) behind `SpeechSynthesizer`. The text frontend and the
/// g2pw polyphone scorer run on ONNX Runtime CPU; calls are serialised because both share
/// sessions and the frontend keeps no per-call state worth locking separately.
final class PrimeSynthesizer: SpeechSynthesizer, @unchecked Sendable {
    private let tts: OrtLite
    private let frontend: PrimeFrontend
    private let lock = NSLock()
    var voice: Int64 = 0  // 0 Xinran (f), 1 Anchen (m), 2 Bowen (m)

    init(ttsPath: String, g2pwPath: String) throws {
        tts = try OrtLite(modelPath: ttsPath)
        let g2pwModel = try OrtLite(modelPath: g2pwPath)

        let res = try JSONDecoder().decode(G2PW.Resources.self, from: Data(contentsOf: LocalModels.resource("g2pw.json")))
        let vocab = try String(contentsOf: LocalModels.resource("vocab.txt"), encoding: .utf8)
        let g2pw = G2PW(resources: res, vocab: vocab)
        let g2pen = G2PEn(cmudict: try String(contentsOf: LocalModels.resource("cmudict.txt"), encoding: .utf8),
                          weights: try Data(contentsOf: LocalModels.resource("g2pen.bin")))
        let table = try JSONSerialization.jsonObject(with: Data(contentsOf: LocalModels.resource("symbol_table.json"))) as? [String: Any]
        let symbols = (table?["symbols"] as? [String]) ?? []
        let labelCount = res.labels.count

        frontend = PrimeFrontend(symbols: symbols, g2pw: g2pw, g2pen: g2pen) { ids, position, charIds, masks in
            let batch = ids.count
            let width = ids.map(\.count).max() ?? 0
            var inputIds = [Int64](), attention = [Int64]()
            for row in ids {
                inputIds += row + [Int64](repeating: 0, count: width - row.count)
                attention += [Int64](repeating: 1, count: row.count) + [Int64](repeating: 0, count: width - row.count)
            }
            let shape = [Int64(batch), Int64(width)]
            let probs = try g2pwModel.run([
                ("input_ids", .int64(inputIds, shape: shape)),
                ("token_type_ids", .int64([Int64](repeating: 0, count: batch * width), shape: shape)),
                ("attention_mask", .int64(attention, shape: shape)),
                ("phoneme_mask", .float(masks.flatMap { $0 }, shape: [Int64(batch), Int64(labelCount)])),
                ("char_ids", .int64(charIds, shape: [Int64(batch)])),
                ("position_ids", .int64(position, shape: [Int64(batch)])),
            ], output: "probs")
            return (0..<batch).map { Array(probs[$0 * labelCount..<($0 + 1) * labelCount]) }
        }
    }

    func synthesize(_ text: String) async throws -> [Float] {
        do { return try await run(text) } catch {
            print("[local] synth failed for '\(text)': \(error)")
            throw error
        }
    }

    private func run(_ text: String) async throws -> [Float] {
        try await Task.detached(priority: .userInitiated) { [self] in
            lock.lock()
            defer { lock.unlock() }
            let out = try frontend.convert(text)
            guard !out.phoneIds.isEmpty else { return [] }
            let blank: ([Int64]) -> [Int64] = { ids in [0] + ids.flatMap { [$0, 0] } }
            let phones = blank(out.phoneIds), tones = blank(out.tones), langs = blank(out.langs)
            let shape = [Int64(1), Int64(phones.count)]
            return try tts.run([
                ("x", .int64(phones, shape: shape)), ("tone", .int64(tones, shape: shape)),
                ("lang", .int64(langs, shape: shape)), ("x_lengths", .int64([Int64(phones.count)], shape: [1])),
                ("sid", .int64([voice], shape: [1])), ("noise_scale", .float([0.667], shape: [1])),
                ("length_scale", .float([1.0], shape: [1])),
            ], output: "wav")
        }.value
    }
}
