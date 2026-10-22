import Foundation

struct Turn: Sendable {
    let user: String
    let assistant: String
}

protocol ResponseGenerator: Sendable {
    func stream(history: [Turn], user: String) -> AsyncThrowingStream<String, Error>
}

protocol SpeechSynthesizer: Sendable {
    /// Mono samples at the playback rate.
    func synthesize(_ text: String) async throws -> [Float]
}

protocol PlaybackSink: AnyObject, Sendable {
    /// Queues audio; `done` fires when it has been played. It must not fire for flushed audio.
    func enqueue(_ samples: [Float], done: @escaping @Sendable () -> Void)
    /// Drops everything queued or playing (barge-in).
    func flush()
}

/// Answer -> speech, with interruption. The text comes from `generator`, is cut into sentences,
/// synthesised one by one and queued for playback. `interrupt` is the single cancellation path,
/// used by voice activity and by an explicit cancel alike.
actor LocalPipeline {
    enum Event: Sendable, Equatable {
        case started(generation: Int)
        case text(String)
        case done(cancelled: Bool)
    }

    nonisolated let scope = CancelScope()
    private let generator: ResponseGenerator
    private let synthesizer: SpeechSynthesizer
    private let sink: PlaybackSink
    private let onEvent: @Sendable (Event) -> Void

    private(set) var history: [Turn] = []
    private var interruptResponse = true
    private var llmTask: Task<Void, Never>?
    private var synthTask: Task<Void, Never>?
    private var user = ""
    private var spoken = ""
    private var outstanding = 0
    private var synthesisDone = false

    init(generator: ResponseGenerator, synthesizer: SpeechSynthesizer, sink: PlaybackSink,
         onEvent: @escaping @Sendable (Event) -> Void) {
        self.generator = generator
        self.synthesizer = synthesizer
        self.sink = sink
        self.onEvent = onEvent
    }

    /// Mirrors `turn_detection.interrupt_response`: when false, speech is still heard but
    /// never stops the answer.
    func setInterruptResponse(_ enabled: Bool) { interruptResponse = enabled }

    @discardableResult
    func respond(to text: String) -> Int {
        if scope.isActive { _ = interrupt(reason: .client) }
        let g = scope.begin()
        user = text
        spoken = ""
        outstanding = 0
        synthesisDone = false
        onEvent(.started(generation: g))

        let past = history
        let (sentences, continuation) = AsyncStream<String>.makeStream()
        let generator = generator, synthesizer = synthesizer, scope = scope

        llmTask = Task {
            var chunker = SentenceChunker()
            do {
                for try await token in generator.stream(history: past, user: text) {
                    if scope.isStale(g) || Task.isCancelled { break }
                    for sentence in chunker.feed(token) { continuation.yield(sentence) }
                }
                if !scope.isStale(g), let tail = chunker.flush() { continuation.yield(tail) }
            } catch {}
            continuation.finish()
        }
        synthTask = Task {
            for await sentence in sentences {
                if scope.isStale(g) { break }
                guard let samples = try? await synthesizer.synthesize(sentence) else { continue }
                await self.enqueue(samples, text: sentence, generation: g)
            }
            await self.synthesisFinished(generation: g)
        }
        return g
    }

    enum Reason { case vad, client }

    /// Stops the current answer. Returns false when there was nothing to stop, or when voice
    /// activity is not allowed to interrupt.
    @discardableResult
    func interrupt(reason: Reason) -> Bool {
        if reason == .vad && !interruptResponse { return false }
        guard scope.cancel() else { return false }
        llmTask?.cancel()
        synthTask?.cancel()
        llmTask = nil
        synthTask = nil
        sink.flush()
        onEvent(.done(cancelled: true))
        return true
    }

    /// Call when the user starts speaking.
    @discardableResult
    func userSpeechStarted() -> Bool { interrupt(reason: .vad) }

    // MARK: - actor-serialised steps (each re-checks staleness, so an interrupt always wins)

    private func enqueue(_ samples: [Float], text: String, generation g: Int) {
        guard !scope.isStale(g) else { return }
        outstanding += 1
        spoken += text
        onEvent(.text(text))
        sink.enqueue(samples) { [weak self] in
            Task { await self?.played(generation: g) }
        }
    }

    private func played(generation g: Int) {
        guard !scope.isStale(g) else { return }
        outstanding -= 1
        completeIfFinished(g)
    }

    private func synthesisFinished(generation g: Int) {
        guard !scope.isStale(g) else { return }
        synthesisDone = true
        completeIfFinished(g)
    }

    private func completeIfFinished(_ g: Int) {
        guard synthesisDone, outstanding == 0 else { return }
        scope.finish(g)
        history.append(Turn(user: user, assistant: spoken))  // interrupted turns are never recorded
        onEvent(.done(cancelled: false))
    }
}
