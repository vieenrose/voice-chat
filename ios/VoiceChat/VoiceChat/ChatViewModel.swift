import AVFoundation
import Foundation

@MainActor
final class ChatViewModel: ObservableObject {
    enum Role { case user, assistant }

    struct Line: Identifiable {
        let id = UUID()
        let role: Role
        var text: String
    }

    @Published private(set) var lines: [Line] = []
    @Published private(set) var status = "未連線"
    @Published private(set) var isRunning = false
    /// End of user speech (server VAD) to first assistant audio received, in ms.
    @Published private(set) var lastLatencyMs: Int?

    private let client = RealtimeClient()
    private let audio = AudioIO()
    private var assistantLine: Int?
    private var speechStoppedAt: Date?

    func start(url urlString: String) async {
        guard let url = URL(string: urlString.trimmingCharacters(in: .whitespaces)) else {
            status = "網址無效"
            return
        }
        guard await AVAudioApplication.requestRecordPermission() else {
            status = "需要麥克風權限"
            return
        }

        client.onEvent = { [weak self] event in self?.handle(event) }
        audio.onCapture = { [client] pcm in client.sendAudio(pcm) }
        status = "連線中…"
        client.connect(to: url)
        do {
            try audio.start()
            isRunning = true
        } catch {
            client.disconnect()
            status = "音訊錯誤：\(error.localizedDescription)"
        }
    }

    func stop() {
        audio.stop()
        client.disconnect()
        isRunning = false
        status = "未連線"
    }

    private func handle(_ event: RealtimeClient.Event) {
        switch event {
        case .connected:
            status = "請說話"
        case .speechStarted:
            // Barge-in: the server cancels the response; drop what is already queued here.
            audio.flushPlayback()
            speechStoppedAt = nil
            status = "聆聽中…"
        case .speechStopped:
            speechStoppedAt = Date()
            status = "思考中…"
        case .userTranscript(let text):
            guard !text.isEmpty else { return }
            lines.append(Line(role: .user, text: ZhTW.convert(text)))
        case .responseStarted:
            lines.append(Line(role: .assistant, text: ""))
            assistantLine = lines.count - 1
        case .assistantText(let chunk):
            guard let index = assistantLine, lines.indices.contains(index) else { return }
            lines[index].text += ZhTW.convert(chunk)
        case .assistantAudio(let pcm):
            if let start = speechStoppedAt {
                lastLatencyMs = Int(Date().timeIntervalSince(start) * 1000)
                speechStoppedAt = nil
                status = "回答中…"
            }
            audio.play(pcm16: pcm)
        case .responseDone:
            if let index = assistantLine, lines.indices.contains(index), lines[index].text.isEmpty {
                lines.remove(at: index)
            }
            assistantLine = nil
            status = "請說話"
        case .error(let message):
            status = "錯誤：\(message)"
        case .closed(let reason):
            if isRunning {
                audio.stop()
                isRunning = false
                status = "連線中斷" + (reason.map { "：\($0)" } ?? "")
            }
        }
    }
}
