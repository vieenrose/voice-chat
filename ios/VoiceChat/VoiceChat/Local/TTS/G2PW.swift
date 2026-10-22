import Foundation

/// Mandarin polyphone resolution with the g2pw BERT model (int8 ONNX), a port of the g2pw package.
/// Characters with a single reading come from a dictionary; the rest are scored in one batched call.
final class G2PW {
    struct Resources: Decodable {
        let labels: [String]
        let chars: [String]
        let char2ph: [String: [Int]]
        let single: [String: String]
    }

    /// Scores a batch of polyphone queries; returns one probability row (labels.count floats) each.
    typealias Scorer = (_ inputIds: [[Int64]], _ position: [Int64], _ charIds: [Int64], _ masks: [[Float]]) throws -> [[Float]]

    struct Query {
        let sentence: Int
        let index: Int          // character index in the sentence
        let inputIds: [Int64]
        let position: Int64
        let charId: Int64
        let allowed: [Int]
    }

    let res: Resources
    let tokenizer: BertTokenizer
    private let charIndex: [String: Int]
    static let window = 32

    init(resources: Resources, vocab: String) {
        res = resources
        tokenizer = BertTokenizer(vocab: vocab)
        charIndex = Dictionary(uniqueKeysWithValues: resources.chars.enumerated().map { ($1, $0) })
    }

    /// Per-character bopomofo (nil for non-Chinese characters) for one sentence.
    func convert(_ sentence: String, score: Scorer) throws -> [String?] {
        let chars = sentence.unicodeScalars.map { String($0) }
        var result = [String?](repeating: nil, count: chars.count)
        var queries: [Query] = []
        for (i, ch) in chars.enumerated() {
            if charIndex[ch] != nil {
                if let q = makeQuery(chars: chars, index: i) { queries.append(q) }
            } else if let reading = res.single[ch] {
                result[i] = reading
            }
        }
        guard !queries.isEmpty else { return result }
        let size = res.labels.count
        let width = queries.map(\.inputIds.count).max() ?? 0
        let ids = queries.map { $0.inputIds + [Int64](repeating: 0, count: width - $0.inputIds.count) }
        let masks = queries.map { q -> [Float] in
            var m = [Float](repeating: 0, count: size)
            for k in q.allowed { m[k] = 1 }
            return m
        }
        let probs = try score(ids, queries.map(\.position), queries.map(\.charId), masks)
        for (q, row) in zip(queries, probs) {
            var best = 0
            for k in 1..<row.count where row[k] > row[best] { best = k }
            result[q.index] = res.labels[best]
        }
        return result
    }

    func makeQuery(chars: [String], index: Int) -> Query? {
        // Window of 32 characters around the query, then lowercase (as TextDataset does).
        let start = max(0, index - Self.window / 2), end = min(chars.count, index + Self.window / 2)
        let text = chars[start..<end].map { $0.lowercased() }
        let q = index - start
        let (tokens, text2token) = tokenizer.tokenizeAndMap(text)
        guard q < text2token.count, let tokenPos = text2token[q] else { return nil }
        let ids = [BertTokenizer.cls] + tokens.map { tokenizer.id($0) } + [BertTokenizer.sep]
        guard let charId = charIndex[chars[index]], let allowed = res.char2ph[chars[index]] else { return nil }
        return Query(sentence: 0, index: index, inputIds: ids.map(Int64.init), position: Int64(tokenPos + 1),
                     charId: Int64(charId), allowed: allowed)
    }
}

/// bert-base-chinese WordPiece, restricted to what g2pw feeds it: one English/number run or one character at a time.
final class BertTokenizer {
    static let cls = 101, sep = 102, unk = 100
    private var vocab: [String: Int] = [:]

    init(vocab text: String) {
        for (i, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() where !line.isEmpty {
            vocab[String(line)] = i
        }
    }

    func id(_ token: String) -> Int { vocab[token] ?? Self.unk }

    private func wordpiece(_ word: String) -> [String] {
        if vocab[word] != nil { return [word] }
        var pieces: [String] = []
        let chars = Array(word)
        var start = 0
        while start < chars.count {
            var end = chars.count
            var found: String?
            while start < end {
                let sub = (start > 0 ? "##" : "") + String(chars[start..<end])
                if vocab[sub] != nil { found = sub; break }
                end -= 1
            }
            guard let piece = found else { return ["[UNK]"] }
            pieces.append(piece)
            start = end
        }
        return pieces
    }

    private static func stripAccents(_ s: String) -> String {
        String(String.UnicodeScalarView(s.decomposedStringWithCanonicalMapping.unicodeScalars.filter { $0.properties.generalCategory != .nonspacingMark }))
    }

    /// g2pw `tokenize_and_map`: tokens plus, for each character, the token it belongs to (nil for spaces).
    func tokenizeAndMap(_ chars: [String]) -> (tokens: [String], textToToken: [Int?]) {
        // wordize: spaces are skipped, [a-zA-Z0-9] runs are one word, anything else is one character.
        var words: [(String, Int, Int)] = []
        var textToWord = [Int?](repeating: nil, count: chars.count)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c == " " { i += 1; continue }
            var j = i + 1
            if c.unicodeScalars.allSatisfy({ $0.isASCII && ($0.properties.isAlphabetic || ("0"..."9").contains(Character($0))) }) {
                while j < chars.count, chars[j].unicodeScalars.allSatisfy({ $0.isASCII && ($0.properties.isAlphabetic || ("0"..."9").contains(Character($0))) }) { j += 1 }
            }
            for k in i..<j { textToWord[k] = words.count }
            words.append((chars[i..<j].joined(), i, j))
            i = j
        }
        var tokens: [String] = []
        var spans: [(Int, Int)] = []
        for (word, s, e) in words {
            let pieces = wordpiece(Self.stripAccents(word).lowercased())
            if pieces.isEmpty || pieces == ["[UNK]"] {
                spans.append((s, e)); tokens.append("[UNK]")
            } else {
                var cursor = s
                for p in pieces {
                    let len = p.hasPrefix("##") ? p.count - 2 : p.count
                    spans.append((cursor, cursor + len)); tokens.append(p)
                    cursor += len
                }
            }
        }
        var map = textToWord
        for (t, span) in spans.enumerated() { for k in span.0..<min(span.1, chars.count) { map[k] = t } }
        return (tokens, map)
    }
}
