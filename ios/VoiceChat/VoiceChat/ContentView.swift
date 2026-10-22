import SwiftUI

struct ContentView: View {
    @StateObject private var model = ChatViewModel()
    @StateObject private var local = LocalChatViewModel()
    /// 伺服器: Realtime server on the Mac. 本機: everything on the phone.
    @AppStorage("localMode") private var localMode = true
    /// The Mac's speech-to-speech server, reached through `tailscale serve` (tailnet only).
    @AppStorage("serverURL") private var serverURL = LocalConfig.serverURL

    private var running: (status: String, latency: Int?, lines: [ChatViewModel.Line], isRunning: Bool) {
        localMode ? (local.status, local.lastLatencyMs, local.lines, local.isRunning)
                  : (model.status, model.lastLatencyMs, model.lines, model.isRunning)
    }

    var body: some View {
        VStack(spacing: 12) {
            Picker("模式", selection: $localMode) {
                Text("本機").tag(true)
                Text("伺服器").tag(false)
            }
            .pickerStyle(.segmented)
            .disabled(model.isRunning || local.isRunning)

            if !localMode {
            TextField("伺服器網址", text: $serverURL)
                .textFieldStyle(.roundedBorder)
                .font(.footnote.monospaced())
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .disabled(model.isRunning)
            }

            HStack {
                Text(running.status)
                Spacer()
                if let ms = running.latency {
                    Text("延遲 \(ms) ms").monospacedDigit()
                }
            }
            .font(.subheadline)
            .foregroundStyle(.secondary)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(running.lines) { line in
                            Bubble(line: line).id(line.id)
                        }
                    }
                }
                .onChange(of: running.lines.last?.text) {
                    if let id = running.lines.last?.id {
                        withAnimation { proxy.scrollTo(id, anchor: .bottom) }
                    }
                }
            }

            Button {
                if running.isRunning {
                    localMode ? local.stop() : model.stop()
                } else {
                    Task { await (localMode ? local.start() : model.start(url: serverURL)) }
                }
            } label: {
                Label(running.isRunning ? "結束對話" : "開始對話",
                      systemImage: running.isRunning ? "stop.circle.fill" : "mic.circle.fill")
                    .font(.title2)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(running.isRunning ? .red : .accentColor)
        }
        .padding()
        .task {
            if ProcessInfo.processInfo.arguments.contains("-selftest") {
                localMode = true
                await local.start()
            }
        }
    }
}

private struct Bubble: View {
    let line: ChatViewModel.Line

    var body: some View {
        let isUser = line.role == .user
        HStack {
            if isUser { Spacer(minLength: 40) }
            Text(line.text.isEmpty ? "…" : line.text)
                .padding(10)
                .background(isUser ? Color.accentColor.opacity(0.2) : Color.secondary.opacity(0.15),
                            in: RoundedRectangle(cornerRadius: 12))
            if !isUser { Spacer(minLength: 40) }
        }
    }
}
