import Foundation

/// Number spelling used by the text normalizer: Chinese (cn2an "low" mode) and English (inflect).
enum NumberWords {
    private static let zhDigits = Array("零一二三四五六七八九").map(String.init)

    static func digitsZh(_ digits: String) -> String { digits.compactMap { $0.wholeNumberValue }.map { zhDigits[$0] }.joined() }

    private static let enDigits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    static func digitsEn(_ digits: String) -> String { digits.compactMap { $0.wholeNumberValue }.map { enDigits[$0] }.joined(separator: " ") }

    // MARK: Chinese cardinal

    /// 0..9999 inside a 万/亿 group; `pad` forces the 零 for a leading zero digit.
    private static func group(_ n: Int) -> String {
        let d = [n / 1000, n / 100 % 10, n / 10 % 10, n % 10]
        let units = ["千", "百", "十", ""]
        var out = ""
        var pendingZero = false
        for (i, v) in d.enumerated() {
            if v == 0 { if !out.isEmpty { pendingZero = true }; continue }
            if pendingZero { out += "零"; pendingZero = false }
            out += zhDigits[v] + units[i]
        }
        return out
    }

    /// cn2an.an2cn(n, "low"): 10 -> 十, 110 -> 一百一十, 10001 -> 一万零一, 2 -> 二.
    static func cardinalZh(_ value: Int) -> String {
        if value == 0 { return "零" }
        let yi = value / 100_000_000, wan = value / 10_000 % 10_000, ge = value % 10_000
        var parts: [(Int, String)] = []
        if yi > 0 { parts.append((yi, "亿")) }
        if wan > 0 { parts.append((wan, "万")) }
        if ge > 0 { parts.append((ge, "")) }
        var out = ""
        for (i, part) in parts.enumerated() {
            var text = group(part.0)
            // a lower group below 1000 after a higher one needs a 零 bridge (一万零一)
            if i > 0 && part.0 < 1000 { text = "零" + text }
            out += text + part.1
        }
        if out.hasPrefix("一十") { out.removeFirst() }
        return out
    }

    static func cardinalZh(_ digits: String) -> String {
        if let n = Int(digits) { return cardinalZh(n) }
        return digitsZh(digits)
    }

    // MARK: English cardinal / ordinal

    private static let small = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    private static let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    private static func below1000(_ n: Int, and: Bool = false) -> [String] {
        var words: [String] = []
        if n >= 100 { words += [small[n / 100], "hundred"] }
        let r = n % 100
        if and && n >= 100 && r > 0 { words.append("and") }
        if r >= 20 {
            words.append(tens[r / 10])
            if r % 10 > 0 { words.append(small[r % 10]) }
        } else if r > 0 {
            words.append(small[r])
        }
        return words
    }

    /// inflect.number_to_words(n, andword=""), hyphens and commas flattened to spaces.
    static func cardinalEn(_ value: Int, and: Bool = false) -> String {
        if value == 0 { return "zero" }
        var words: [String] = []
        var rest = value
        let scales = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]
        for (size, name) in scales where rest >= size {
            words += below1000(rest / size, and: and) + [name]
            rest %= size
        }
        // inflect puts "and" before a final group under 100 that follows a larger one (one thousand and first)
        if and && rest > 0 && rest < 100 && value >= 100 { words.append("and") }
        words += below1000(rest, and: and)
        return words.joined(separator: " ")
    }

    static func cardinalEn(_ digits: String) -> String {
        if let n = Int(digits) { return cardinalEn(n) }
        return digitsEn(digits)
    }

    static func ordinalEn(_ value: Int) -> String {
        var words = cardinalEn(value, and: true).split(separator: " ").map(String.init)
        guard let last = words.popLast() else { return "" }
        let irregular = ["one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                         "nine": "ninth", "twelve": "twelfth"]
        let word: String
        if let special = irregular[last] {
            word = special
        } else if last.hasSuffix("y") {
            word = last.dropLast() + "ieth"
        } else {
            word = last + "th"
        }
        return (words + [word]).joined(separator: " ")
    }

    static func ordinalEn(_ digits: String) -> String {
        if let n = Int(digits) { return ordinalEn(n) }
        return digitsEn(digits)
    }
}
