import AVFoundation
import Foundation
import HuggingFace
import MLX
import MLXLLM
import MLXLMCommon
import UIKit

/// Fully on-device conversation: X-ASR -> Bonsai 4B -> PrimeTTS, with barge-in on partial text.
/// `-selftest` on the launch arguments feeds a wav instead of the microphone and prints the
/// stage timings to the console, so latency can be measured without anybody speaking.
@MainActor
final class LocalChatViewModel: ObservableObject {
    @Published private(set) var lines: [ChatViewModel.Line] = []
    @Published private(set) var status = "尚未載入"
    @Published private(set) var isRunning = false
    @Published private(set) var lastLatencyMs: Int?

    private let audio = AudioIO()
    private var asr: LocalASR?
    private var pipeline: LocalPipeline?
    private var generator: BonsaiGenerator?
    private var assistantLine: Int?
    private var speechEndedAt: Date?
    private var endpointAt: Date?
    private var firstTokenSecs: Double?
    private var pipelineActive = false
    private var loaded = false

    private func log(_ s: String) { print("[local] \(s)") }

    private func load() async throws {
        guard !loaded else { return }
        Memory.cacheLimit = 64 * 1024 * 1024
        status = "載入語音辨識…"
        let asr = try LocalASR()
        status = "下載／載入語音合成…"
        let tts = try await LocalModels.fetch(LocalModels.ttsURL)
        let g2pw = try await LocalModels.fetch(LocalModels.g2pwURL)
        let synth = try PrimeSynthesizer(ttsPath: tts, g2pwPath: g2pw)
        status = "下載／載入語言模型…"
        let container = try await loadModelContainer(
            from: HubDownloader(), using: TransformersLoader(), id: LocalModels.llmRepo
        ) { [weak self] p in
            Task { @MainActor in self?.status = String(format: "下載語言模型 %.0f%%", p.fractionCompleted * 100) }
        }
        let generator = BonsaiGenerator(container: container)
        generator.onFirstToken = { [weak self] secs in Task { @MainActor in self?.firstTokenSecs = secs } }

        let sink = AudioIOSink(audio)
        pipeline = LocalPipeline(generator: generator, synthesizer: synth, sink: sink) { [weak self] event in
            Task { @MainActor in self?.handle(event) }
        }
        asr.onPartial = { [weak self] text in Task { @MainActor in self?.partial(text) } }
        asr.onEndpoint = { [weak self] text in Task { @MainActor in self?.endpoint(text) } }
        self.asr = asr
        self.generator = generator
        loaded = true
    }

    func start() async {
        guard await AVAudioApplication.requestRecordPermission() else {
            status = "需要麥克風權限"
            return
        }
        do {
            try await load()
            audio.onCapture = { [asr] pcm in asr?.feed(pcm16: pcm) }
            try audio.start()
            UIApplication.shared.isIdleTimerDisabled = true
            isRunning = true
            status = "請說話"
            if ProcessInfo.processInfo.arguments.contains("-selftest") {
                Task { await selfTest() }
            }
        } catch {
            status = "錯誤：\(error.localizedDescription)"
            log("start failed: \(error)")
        }
    }

    func stop() {
        audio.stop()
        Task { await pipeline?.interrupt(reason: .client) }
        UIApplication.shared.isIdleTimerDisabled = false
        isRunning = false
        status = "已停止"
    }

    // MARK: - ASR events

    private func partial(_ text: String) {
        status = "聆聽中…"
        // Barge-in: enough recognised text while the assistant is talking.
        if pipelineActive, text.count >= 3 {
            log("barge-in on partial '\(text)'")
            Task { await pipeline?.userSpeechStarted() }
        }
    }

    private func endpoint(_ text: String) {
        endpointAt = Date()
        if let end = speechEndedAt { log(String(format: "endpoint %.2f s after end of speech: %@", Date().timeIntervalSince(end), text)) }
        lines.append(.init(role: .user, text: ZhTW.convert(text)))
        status = "思考中…"
        firstTokenSecs = nil
        Task { await pipeline?.respond(to: text) }
    }

    // MARK: - pipeline events

    private func handle(_ event: LocalPipeline.Event) {
        switch event {
        case .started:
            pipelineActive = true
            lines.append(.init(role: .assistant, text: ""))
            assistantLine = lines.count - 1
        case .text(let chunk):
            if let start = speechEndedAt ?? endpointAt {
                let ms = Int(Date().timeIntervalSince(start) * 1000)
                lastLatencyMs = ms
                log("first audio queued \(ms) ms after end of speech (LLM first token \(firstTokenSecs.map { String(format: "%.2f s", $0) } ?? "?"))")
                speechEndedAt = nil
                endpointAt = nil
            }
            status = "回答中…"
            if let i = assistantLine, lines.indices.contains(i) { lines[i].text += ZhTW.convert(chunk) }
        case .done(let cancelled):
            pipelineActive = false
            if let i = assistantLine, lines.indices.contains(i), lines[i].text.isEmpty { lines.remove(at: i) }
            assistantLine = nil
            if isRunning { status = "請說話" }
            log("done cancelled=\(cancelled)")
        }
    }

    // MARK: - self test (wav in place of the microphone)

    private func wavSamples(_ rel: String) throws -> [Float] {
        let file = try AVAudioFile(forReading: LocalModels.asrFile(rel))
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else { return [] }
        try file.read(into: buffer)
        return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    }

    /// Streams a wav in real time (160 ms chunks) followed by silence, like a microphone would.
    private func inject(_ samples: [Float], silence: Double = 1.5) async {
        let chunk = 2560
        let padded = samples + [Float](repeating: 0, count: Int(silence * 16_000))
        var i = 0
        while i < padded.count {
            asr?.feed(Array(padded[i..<min(i + chunk, padded.count)]))
            i += chunk
            if i >= samples.count, speechEndedAt == nil, endpointAt == nil { speechEndedAt = Date() }
            try? await Task.sleep(for: .milliseconds(160))
        }
    }

    private func selfTest() async {
        do {
            log("selftest: turn 1")
            speechEndedAt = nil
            let first = try wavSamples("test_wavs/0.wav")
            // mark end of speech at the moment the last real sample is fed
            await inject(first)
            try? await Task.sleep(for: .seconds(8))
            log("selftest: turn 2 with barge-in")
            let second = try wavSamples("test_wavs/1.wav")
            await inject(second, silence: 0.3)
            try? await Task.sleep(for: .seconds(1.5))
            log("selftest: injecting interruption")
            speechEndedAt = nil
            await inject(try wavSamples("test_wavs/2.wav"), silence: 1.5)
            try? await Task.sleep(for: .seconds(6))
            log("selftest finished; footprint \(Int(Double(os_proc_available_memory()) / 1_048_576)) MB headroom")
        } catch {
            log("selftest failed: \(error)")
        }
    }
}
