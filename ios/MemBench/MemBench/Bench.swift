import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import AVFoundation
import SherpaOnnxShared
import MLXLLM
import MLXLMCommon
import Tokenizers
import os

/// Memory probe for the fully local stack on a 6 GB iPhone. Loads ASR, TTS and LLM one after
/// the other and logs the process footprint (what jetsam looks at) after each step.
@MainActor
final class Bench: ObservableObject {
    @Published var log = ""
    @Published var running = false

    let asrBase = "https://huggingface.co/csukuangfj2/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/resolve/main/"
    let ttsURL = URL(string: "https://huggingface.co/Luigi/PrimeTTS/resolve/main/v21_mbistft_16k/primetts_v21_3voice.onnx")!
    let g2pwURL = URL(string: "https://huggingface.co/Luigi/PrimeTTS/resolve/main/g2pw_int8/g2pw_int8.onnx")!
    var llmRepo = "prism-ml/Ternary-Bonsai-4B-mlx-2bit"

    private var asr: SherpaOnnxRecognizer?
    private var tts: OrtLite?
    private var g2pw: OrtLite?
    private var llm: ModelContainer?

    /// Physical footprint of this process in MB.
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    func note(_ label: String) {
        let avail = Double(os_proc_available_memory()) / 1_048_576
        let mlx = Double(Memory.activeMemory) / 1_048_576
        let peak = Double(Memory.peakMemory) / 1_048_576
        log += String(format: "%@: footprint %.0f MB, MLX active %.0f / peak %.0f MB, jetsam headroom %.0f MB\n",
                      label, Self.footprintMB(), mlx, peak, avail)
    }

    func timed<T>(_ label: String, _ body: () async throws -> T) async -> T? {
        let t = Date()
        do {
            let value = try await body()
            log += String(format: "%@ ok in %.1f s\n", label, Date().timeIntervalSince(t))
            note(label)
            return value
        } catch {
            log += "\(label) FAILED: \(error)\n"
            return nil
        }
    }

    func runAll() async {
        guard !running else { return }
        running = true
        defer { running = false }
        log = ""
        Memory.cacheLimit = 64 * 1024 * 1024  // small GPU buffer cache, as recommended on iPhone
        note("start")

        asr = await timed("load X-ASR zipformer (sherpa-onnx)") { try await self.loadASR() }
        tts = await timed("load PrimeTTS v2.1 (ONNX)") { try await self.loadORT(self.ttsURL) }
        g2pw = await timed("load g2pw int8 (ONNX)") { try await self.loadORT(self.g2pwURL) }
        llm = await timed("load LLM \(llmRepo)") {
            try await loadModelContainer(from: HubDownloader(), using: TransformersLoader(), id: llmRepo)
        }

        if let llm {
            _ = await timed("LLM generate") {
                let session = ChatSession(llm, generateParameters: GenerateParameters(maxTokens: 60, temperature: 0.7))
                let reply = try await session.respond(to: "用一句話介紹台灣。")
                log += "  reply: \(reply)\n"
            }
        }
        if let tts {
            _ = await timed("PrimeTTS synth (dummy 40 phones)") { try self.synthDummy(tts) }
        }
        if asr != nil {
            _ = await timed("ASR transcribe test wavs") { try await self.transcribeTests() }
        }
        log += "done\n"
    }

    /// Downloads a model file to Caches once, then opens it with ONNX Runtime (CPU).
    func loadORT(_ url: URL) async throws -> OrtLite {
        let dest = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(url.lastPathComponent)
        if !FileManager.default.fileExists(atPath: dest.path) {
            let (tmp, _) = try await URLSession.shared.download(from: url)
            try FileManager.default.moveItem(at: tmp, to: dest)
        }
        return try OrtLite(modelPath: dest.path)
    }

    /// Runs the 3-voice VITS on made-up ids: memory and speed do not depend on the content.
    func synthDummy(_ session: OrtLite) throws {
        let n = 40
        let blank: ([Int64]) -> [Int64] = { ids in [0] + ids.flatMap { [$0, 0] } }
        let phones = blank((0..<n).map { Int64(4 + $0 % 36) })
        let tones = blank((0..<n).map { Int64(1 + $0 % 5) })
        let langs = blank(Array(repeating: 0, count: n))
        let shape = [Int64(1), Int64(phones.count)]
        let t = Date()
        let wav = try session.run([
            ("x", .int64(phones, shape: shape)), ("tone", .int64(tones, shape: shape)),
            ("lang", .int64(langs, shape: shape)), ("x_lengths", .int64([Int64(phones.count)], shape: [1])),
            ("sid", .int64([0], shape: [1])), ("noise_scale", .float([0.667], shape: [1])),
            ("length_scale", .float([1.0], shape: [1])),
        ], output: "wav")
        let audio = Double(wav.count) / 16_000
        let took = Date().timeIntervalSince(t)
        log += String(format: "  %d samples = %.2f s audio in %.2f s (RTF %.2f)\n", wav.count, audio, took, took / audio)
    }

    /// The 160 ms streaming X-ASR files are pushed into Caches/xasr by devicectl (no download).
    func asrFile(_ rel: String) throws -> URL {
        let url = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("xasr/" + rel)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NSError(domain: "Bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing \(url.path)"])
        }
        return url
    }

    func loadASR() async throws -> SherpaOnnxRecognizer {
        let enc = try asrFile("encoder.int8.onnx"), dec = try asrFile("decoder.onnx")
        let join = try asrFile("joiner.int8.onnx"), tokens = try asrFile("tokens.txt")
        let transducer = sherpaOnnxOnlineTransducerModelConfig(encoder: enc.path, decoder: dec.path, joiner: join.path)
        let model = sherpaOnnxOnlineModelConfig(tokens: tokens.path, transducer: transducer, numThreads: 2)
        var config = sherpaOnnxOnlineRecognizerConfig(
            featConfig: sherpaOnnxFeatureConfig(sampleRate: 16000, featureDim: 80), modelConfig: model)
        return SherpaOnnxRecognizer(config: &config)
    }

    /// Streams each test wav in 160 ms chunks, as the mic would, and logs text and speed.
    func transcribeTests() async throws {
        guard let asr else { return }
        for i in 0..<4 {
            let file = try AVAudioFile(forReading: try asrFile("test_wavs/\(i).wav"))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else { continue }
            try file.read(into: buffer)
            let samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
            let rate = Int(file.processingFormat.sampleRate)
            let chunk = rate * 160 / 1000
            let t = Date()
            var firstText: Double?
            var start = 0
            while start < samples.count {
                asr.acceptWaveform(samples: Array(samples[start..<min(start + chunk, samples.count)]), sampleRate: rate)
                while asr.isReady() { asr.decode() }
                if firstText == nil, !asr.getResult().text.isEmpty { firstText = Date().timeIntervalSince(t) }
                start += chunk
            }
            asr.acceptWaveform(samples: [Float](repeating: 0, count: rate / 2), sampleRate: rate)
            asr.inputFinished()
            while asr.isReady() { asr.decode() }
            let text = asr.getResult().text
            let audio = Double(samples.count) / Double(rate)
            let took = Date().timeIntervalSince(t)
            log += String(format: "  wav %d (%.1f s, RTF %.2f, first text after %.2f s compute): %@\n", i, audio, took / audio, firstText ?? -1, text)
            asr.reset()
        }
    }
}
