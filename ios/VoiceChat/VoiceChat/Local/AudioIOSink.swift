import Foundation

/// Plays the pipeline's audio on the phone speaker (echo-cancelled by AudioIO's voice processing).
final class AudioIOSink: PlaybackSink, @unchecked Sendable {
    private let audio: AudioIO
    init(_ audio: AudioIO) { self.audio = audio }

    func enqueue(_ samples: [Float], done: @escaping @Sendable () -> Void) { audio.play(samples: samples, done: done) }
    func flush() { audio.flushPlayback() }
}
