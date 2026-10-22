import SwiftUI

struct BenchView: View {
    @StateObject private var bench = Bench()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Button(bench.running ? "Running…" : "Run memory bench") {
                Task { await bench.runAll() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(bench.running)
            ScrollView {
                Text(bench.log)
                    .font(.system(.footnote, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
        }
        .padding()
    }
}
