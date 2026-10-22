import Foundation
import SherpaOnnxShared

/// Streaming X-ASR (sherpa-onnx online transducer, 160 ms chunks). Feed it the mic's 16 kHz
/// PCM16; it reports growing partial text and, when the endpointer sees trailing silence after
/// speech, the final text of the utterance.
final class LocalASR: @unchecked Sendable {
    var onPartial: (@Sendable (String) -> Void)?
    var onEndpoint: (@Sendable (String) -> Void)?

    private let recognizer: SherpaOnnxRecognizer
    private let queue = DispatchQueue(label: "asr", qos: .userInitiated)
    private var lastPartial = ""

    init() throws {
        let transducer = sherpaOnnxOnlineTransducerModelConfig(
            encoder: try LocalModels.asrFile("encoder.int8.onnx").path,
            decoder: try LocalModels.asrFile("decoder.onnx").path,
            joiner: try LocalModels.asrFile("joiner.int8.onnx").path)
        let model = sherpaOnnxOnlineModelConfig(
            tokens: try LocalModels.asrFile("tokens.txt").path, transducer: transducer, numThreads: 2)
        var config = sherpaOnnxOnlineRecognizerConfig(
            featConfig: sherpaOnnxFeatureConfig(sampleRate: 16000, featureDim: 80), modelConfig: model,
            enableEndpoint: true, rule1MinTrailingSilence: 2.4, rule2MinTrailingSilence: 0.7,
            rule3MinUtteranceLength: 25)
        recognizer = SherpaOnnxRecognizer(config: &config)
    }

    func feed(pcm16 data: Data) {
        let samples: [Float] = data.withUnsafeBytes { raw in
            raw.bindMemory(to: Int16.self).map { Float(Int16(littleEndian: $0)) / 32_768 }
        }
        feed(samples)
    }

    func feed(_ samples: [Float]) {
        queue.async { [self] in
            recognizer.acceptWaveform(samples: samples, sampleRate: 16000)
            while recognizer.isReady() { recognizer.decode() }
            let text = recognizer.getResult().text.trimmingCharacters(in: .whitespaces)
            if recognizer.isEndpoint() {
                recognizer.reset()
                lastPartial = ""
                if !text.isEmpty { onEndpoint?(text) }
            } else if text != lastPartial {
                lastPartial = text
                if !text.isEmpty { onPartial?(text) }
            }
        }
    }
}
