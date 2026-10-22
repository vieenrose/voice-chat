import Foundation

/// Minimal OpenAI Realtime client for the speech-to-speech server.
///
/// The server owns VAD, turn taking, STT, LLM and TTS; the client only streams microphone
/// audio in and plays audio out. Input audio is 16 kHz PCM16 and is deliberately not declared
/// in a `session.update` (the server schema pins 24 kHz there). Events are delivered on the
/// main queue.
final class RealtimeClient {
    enum Event {
        case connected
        case speechStarted
        case speechStopped
        case userTranscript(String)
        case responseStarted
        /// One spoken chunk of assistant text; chunks of a response accumulate.
        case assistantText(String)
        /// PCM16 little-endian mono at 16 kHz (server default).
        case assistantAudio(Data)
        case responseDone
        case error(String)
        case closed(String?)
    }

    var onEvent: ((Event) -> Void)?

    private let session = URLSession(configuration: .default)
    private var task: URLSessionWebSocketTask?

    func connect(to url: URL) {
        disconnect()
        let task = session.webSocketTask(with: url)
        self.task = task
        task.resume()
        receive(on: task)
        // Any frame (including the server's session.created) confirms the socket is open.
    }

    func disconnect() {
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
    }

    /// Thread safe: called from the audio capture thread.
    func sendAudio(_ pcm16: Data) {
        send(["type": "input_audio_buffer.append", "audio": pcm16.base64EncodedString()])
    }

    private func send(_ object: [String: Any]) {
        guard let task, let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8) else { return }
        task.send(.string(text)) { _ in }
    }

    private func receive(on task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self, task === self.task else { return }
            switch result {
            case .success(let message):
                if case .string(let text) = message { self.handle(text) }
                self.receive(on: task)
            case .failure(let error):
                self.emit(.closed(error.localizedDescription))
            }
        }
    }

    private func handle(_ text: String) {
        guard let data = text.data(using: .utf8),
              let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = msg["type"] as? String else { return }

        switch type {
        case "session.created":
            emit(.connected)
        case "input_audio_buffer.speech_started":
            emit(.speechStarted)
        case "input_audio_buffer.speech_stopped":
            emit(.speechStopped)
        case "conversation.item.input_audio_transcription.completed":
            emit(.userTranscript(msg["transcript"] as? String ?? ""))
        case "response.created":
            emit(.responseStarted)
        case "response.output_audio_transcript.done":
            emit(.assistantText(msg["transcript"] as? String ?? ""))
        case "response.output_audio.delta":
            if let b64 = msg["delta"] as? String, let pcm = Data(base64Encoded: b64) {
                emit(.assistantAudio(pcm))
            }
        case "response.done":
            emit(.responseDone)
        case "error":
            let detail = (msg["error"] as? [String: Any])?["message"] as? String
            emit(.error(detail ?? text))
        default:
            break
        }
    }

    private func emit(_ event: Event) {
        DispatchQueue.main.async { [weak self] in self?.onEvent?(event) }
    }
}
