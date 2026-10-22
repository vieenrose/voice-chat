import Foundation

/// Cuts a token stream into speakable pieces. Sentence enders always cut; a comma cuts once
/// the piece is long enough, so the first audio starts early on long sentences.
struct SentenceChunker {
    private var buffer = ""
    private let enders: Set<Character> = ["。", "！", "？", "!", "?", "；", ";", "\n"]
    private let softBreaks: Set<Character> = ["，", ",", "、", "："]
    private let softMin = 12

    mutating func feed(_ token: String) -> [String] {
        var out: [String] = []
        for ch in token {
            buffer.append(ch)
            if enders.contains(ch) || (softBreaks.contains(ch) && buffer.count >= softMin) {
                emit(into: &out)
            }
        }
        return out
    }

    mutating func flush() -> String? {
        var out: [String] = []
        emit(into: &out)
        return out.first
    }

    private mutating func emit(into out: inout [String]) {
        let piece = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer = ""
        if piece.contains(where: { $0.isLetter || $0.isNumber }) { out.append(piece) }
    }
}
