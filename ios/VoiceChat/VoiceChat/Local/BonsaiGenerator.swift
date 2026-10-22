import Foundation
import MLXLLM
import MLXLMCommon

/// Bonsai 4B (MLX ternary) behind `ResponseGenerator`. A fresh `ChatSession` per turn is
/// restored from the recorded history, so interrupted turns (never recorded) leave no trace.
final class BonsaiGenerator: ResponseGenerator, @unchecked Sendable {
    static let system = "你是語音助理。請用台灣繁體中文，口語、簡短地回答，一到兩句話就好。不要使用 Markdown、列表、星號或表情符號。"

    private let container: ModelContainer
    /// Called with the seconds from request to first token (for the latency readout).
    var onFirstToken: (@Sendable (Double) -> Void)?

    init(container: ModelContainer) { self.container = container }

    func stream(history: [Turn], user: String) -> AsyncThrowingStream<String, Error> {
        let container = container, onFirstToken = onFirstToken
        return AsyncThrowingStream { continuation in
            let task = Task {
                var messages: [Chat.Message] = [.system(Self.system)]
                for turn in history.suffix(6) {
                    messages.append(.user(turn.user))
                    messages.append(.assistant(turn.assistant))
                }
                let session = ChatSession(
                    container, history: messages,
                    generateParameters: GenerateParameters(maxTokens: 200, temperature: 0.6),
                    additionalContext: ["enable_thinking": false])
                let t0 = Date()
                var first = true
                do {
                    for try await chunk in session.streamResponse(to: user) {
                        if first {
                            first = false
                            onFirstToken?(Date().timeIntervalSince(t0))
                        }
                        let clean = chunk.filter { !"*#`_".contains($0) }
                        if !clean.isEmpty { continuation.yield(clean) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
