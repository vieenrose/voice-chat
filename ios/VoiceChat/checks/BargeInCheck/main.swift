import Foundation

// Command-line check of the interruption logic with fake LLM / TTS / speaker.
// Build and run (Mac):  ./run_check.sh

struct FakeLLM: ResponseGenerator {
    let counter: Counter
    let reply = "第一句話在這裡。第二句話接著說。第三句話也很長。第四句話繼續。第五句話快結束。第六句話是最後一句。"
    func stream(history: [Turn], user: String) -> AsyncThrowingStream<String, Error> {
        let reply = reply, counter = counter
        return AsyncThrowingStream { c in
            let task = Task {
                for ch in reply {
                    if Task.isCancelled { break }
                    try? await Task.sleep(nanoseconds: 15_000_000)
                    counter.add()
                    c.yield(String(ch))
                }
                c.finish()
            }
            c.onTermination = { _ in task.cancel() }
        }
    }
}

struct FakeTTS: SpeechSynthesizer {
    func synthesize(_ text: String) async throws -> [Float] {
        try await Task.sleep(nanoseconds: 60_000_000)   // not interruptible, like an ONNX session run
        return [Float](repeating: 0.1, count: 1_600)      // 0.1 s at 16 kHz
    }
}

final class Counter: @unchecked Sendable {
    private let lock = NSLock(); private var n = 0
    func add() { lock.withLock { n += 1 } }
    var value: Int { lock.withLock { n } }
}

/// Plays each buffer for its duration (x1), unless flushed.
final class FakeSpeaker: PlaybackSink, @unchecked Sendable {
    private let lock = NSLock()
    private var epoch = 0
    private(set) var enqueued = 0
    private(set) var flushes = 0
    private(set) var enqueuedAfterFlush = 0
    func enqueue(_ samples: [Float], done: @escaping @Sendable () -> Void) {
        let myEpoch: Int = lock.withLock { enqueued += 1; if flushes > 0 { enqueuedAfterFlush += 1 }; return epoch }
        Task {
            try? await Task.sleep(nanoseconds: UInt64(Double(samples.count) / 16_000 * 1e9))
            let stillValid = lock.withLock { epoch == myEpoch }
            if stillValid { done() }
        }
    }
    func flush() { lock.withLock { epoch += 1; flushes += 1 } }
    func reset() { lock.withLock { enqueued = 0; enqueuedAfterFlush = 0; flushes = 0 } }
}

final class Events: @unchecked Sendable {
    private let lock = NSLock(); private var list: [LocalPipeline.Event] = []
    func add(_ e: LocalPipeline.Event) { lock.withLock { list.append(e) } }
    var all: [LocalPipeline.Event] { lock.withLock { list } }
    var dones: [Bool] { all.compactMap { if case .done(let c) = $0 { return c } else { return nil } } }
    var texts: [String] { all.compactMap { if case .text(let t) = $0 { return t } else { return nil } } }
}

var failures = 0
func check(_ ok: Bool, _ what: String) {
    print(ok ? "  ok   " : "  FAIL ", what)
    if !ok { failures += 1 }
}
func ms(_ n: UInt64) async { try? await Task.sleep(nanoseconds: n * 1_000_000) }

func make() -> (LocalPipeline, FakeSpeaker, Events, Counter) {
    let speaker = FakeSpeaker(), events = Events(), counter = Counter()
    let p = LocalPipeline(generator: FakeLLM(counter: counter), synthesizer: FakeTTS(), sink: speaker,
                          onEvent: { events.add($0) })
    return (p, speaker, events, counter)
}

// 1. A normal answer plays to the end and is recorded.
print("1. normal answer")
do {
    let (p, speaker, events, _) = make()
    await p.respond(to: "你好")
    await ms(2500)
    check(events.dones == [false], "finishes once, not cancelled")
    check(events.texts.count == 6, "six sentences spoken (got \(events.texts.count))")
    check(await p.history.count == 1, "turn recorded")
    check(speaker.flushes == 0, "no flush")
}

// 2. Barge-in in the middle: stops everything, nothing stale arrives, turn not recorded.
print("2. barge-in mid-answer")
do {
    let (p, speaker, events, counter) = make()
    await p.respond(to: "你好")
    await ms(500)
    let spokenBefore = events.texts.count
    let tokensBefore = counter.value
    let interrupted = await p.userSpeechStarted()
    check(interrupted, "interrupt accepted")
    check(spokenBefore >= 1 && spokenBefore < 6, "was mid-answer (\(spokenBefore) sentences)")
    await ms(1500)
    check(events.dones == [true], "exactly one done, cancelled (got \(events.dones))")
    check(speaker.flushes == 1, "speaker flushed once")
    check(events.texts.count == spokenBefore, "no sentence after the interrupt (\(events.texts.count) vs \(spokenBefore))")
    check(speaker.enqueuedAfterFlush == 0, "no stale audio reached the speaker")
    check(counter.value - tokensBefore <= 2, "LLM stopped streaming (\(counter.value - tokensBefore) tokens after)")
    check(await p.history.isEmpty, "interrupted turn not recorded")
    check(!p.scope.isActive, "scope idle")
}

// 3. Interrupt with nothing playing is a no-op and does not swallow the next answer.
print("3. interrupt while idle")
do {
    let (p, speaker, events, _) = make()
    let before = p.scope.isStale(0)
    let interrupted = await p.userSpeechStarted()
    check(!interrupted, "no-op")
    check(speaker.flushes == 0, "no flush")
    _ = before
    await p.respond(to: "你好")
    await ms(2500)
    check(events.dones == [false] && events.texts.count == 6, "next answer plays fully")
}

// 4. New answer right after a barge-in is not swallowed by the old one.
print("4. answer after barge-in")
do {
    let (p, speaker, events, _) = make()
    await p.respond(to: "第一個問題")
    await ms(400)
    _ = await p.userSpeechStarted()
    speaker.reset()
    await p.respond(to: "第二個問題")
    await ms(2500)
    check(events.dones == [true, false], "cancelled then completed (got \(events.dones))")
    check(speaker.enqueued == 6, "all six sentences of the new answer reached the speaker (\(speaker.enqueued))")
    check(await p.history.map(\.user) == ["第二個問題"], "only the completed turn recorded")
}

// 5. interrupt_response = false: speech never stops the answer.
print("5. interrupt disabled")
do {
    let (p, speaker, events, _) = make()
    await p.setInterruptResponse(false)
    await p.respond(to: "你好")
    await ms(400)
    let interrupted = await p.userSpeechStarted()
    check(!interrupted, "ignored")
    await ms(2200)
    check(events.dones == [false] && speaker.flushes == 0, "answer completes")
}

// 6. Explicit cancel uses the same path even when interrupt_response is off.
print("6. client cancel")
do {
    let (p, speaker, events, _) = make()
    await p.setInterruptResponse(false)
    await p.respond(to: "你好")
    await ms(400)
    let ok = await p.interrupt(reason: .client)
    await ms(800)
    check(ok && events.dones == [true] && speaker.flushes == 1, "cancelled through the same path")
}

print(failures == 0 ? "ALL OK" : "\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
