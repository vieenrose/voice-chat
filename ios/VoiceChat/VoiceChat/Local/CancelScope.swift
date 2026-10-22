import Foundation

/// One counter decides what is still wanted. Every response takes a generation number, every
/// piece of its output (tokens, sentences, audio) is checked against it, and an interruption
/// just bumps the counter: whatever the old response still has in flight is stale at once.
/// Same idea as the speech-to-speech server's CancelScope.
final class CancelScope: @unchecked Sendable {
    private let lock = NSLock()
    private var generation = 0
    private var active = false

    /// Starts a response and returns its generation.
    func begin() -> Int {
        lock.withLock {
            generation += 1
            active = true
            return generation
        }
    }

    func isStale(_ g: Int) -> Bool { lock.withLock { g != generation } }

    /// True from `begin` until the response finishes playing or is cancelled.
    var isActive: Bool { lock.withLock { active } }

    /// Interrupts the current response. Does nothing when no response is active: otherwise a
    /// cough while the assistant is silent would bump the counter for no reason.
    @discardableResult
    func cancel() -> Bool {
        lock.withLock {
            guard active else { return false }
            generation += 1
            active = false
            return true
        }
    }

    /// Marks generation `g` complete; ignored if it was already superseded.
    func finish(_ g: Int) {
        lock.withLock { if g == generation { active = false } }
    }
}
