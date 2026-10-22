import Foundation

/// zh-TW/en text -> PrimeTTS inputs (phone ids, tones, per-phone language), a port of `frontend_bopomofo.py`.
final class PrimeFrontend {
    struct Output {
        var phones: [String] = []
        var phoneIds: [Int64] = []
        var tones: [Int64] = []
        var langs: [Int64] = []
    }

    private let symbolIds: [String: Int]
    private let g2pw: G2PW
    private let g2pen: G2PEn
    private let score: G2PW.Scorer
    static let punct: Set<Character> = [",", ".", "?", "!", "…", "-", "'"]

    init(symbols: [String], g2pw: G2PW, g2pen: G2PEn, score: @escaping G2PW.Scorer) {
        symbolIds = Dictionary(uniqueKeysWithValues: symbols.enumerated().map { ($1, $0) })
        self.g2pw = g2pw
        self.g2pen = g2pen
        self.score = score
    }

    /// '[ㄓㄨㄢ]3' -> (units, tone); empty-rime syllables get the syllabic vowel ㄭ appended.
    private func split(_ syllable: String) -> ([String], Int) {
        var s = Array(syllable)
        var tone = 0
        if let last = s.last, let d = last.wholeNumberValue { tone = d; s.removeLast() }
        var units = s.map(String.init).filter { symbolIds[$0] != nil }
        if units.count == 1, "ㄓㄔㄕㄖㄗㄘㄙ".contains(units[0]), symbolIds["ㄭ"] != nil { units.append("ㄭ") }
        return (units, tone)
    }

    func convert(_ raw: String) throws -> Output {
        let text = TextNorm.normalize(raw)
        let bopomofo = try g2pw.convert(text, score: score)
        let chars = text.unicodeScalars.map { Character($0) }
        var out = Output()
        func add(_ phone: String, _ tone: Int, _ lang: Int64) {
            out.phones.append(phone)
            out.phoneIds.append(Int64(symbolIds[phone] ?? symbolIds["UNK"] ?? 2))
            out.tones.append(Int64(tone))
            out.langs.append(lang)
        }
        var i = 0
        while i < chars.count {
            let ch = chars[i]
            if i < bopomofo.count, let b = bopomofo[i] {
                let (units, tone) = split(b)
                for u in units { add(u, min(tone, 5), 0) }
                i += 1
            } else if ch.isASCII && ch.isLetter {
                var j = i
                while j < chars.count, chars[j].isASCII, chars[j].isLetter || chars[j] == "'" { j += 1 }
                for p in g2pen.pronounce(String(chars[i..<j])) {
                    var phone = p, stress = 0
                    if let d = p.last?.wholeNumberValue { stress = d; phone.removeLast() }
                    if symbolIds[phone] != nil { add(phone, stress, 1) }
                }
                add("SP", 0, 1)
                i = j
            } else {
                if Self.punct.contains(ch) { add(String(ch), 0, 0) }
                else if ch.isWhitespace, let last = out.phones.last, last != "SP" { add("SP", 0, 0) }
                i += 1
            }
        }
        return out
    }
}
