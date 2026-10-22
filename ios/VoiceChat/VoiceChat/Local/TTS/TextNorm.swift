import Foundation

/// Phone-attendant text normalizer (zh-TW + English): a line-by-line port of PrimeTTS `text_norm.py`.
/// Entities (email, time, date, temperature, percent, price, phone, serial, decimals, counts) become
/// spoken forms; digit-by-digit vs cardinal depends on the entity and the surrounding language.
enum TextNorm {
    private static let addr = Set("號樓段巷弄室坪")
    private static let months = ["January", "February", "March", "April", "May", "June", "July", "August",
                                 "September", "October", "November", "December"]

    // MARK: regex helpers (UTF-16 offsets, like Python's str indices for BMP text)

    struct Hit {
        let text: String
        let groups: [String?]
        let start: Int
        let end: Int
    }

    private static func sub(_ pattern: String, _ text: String, ignoreCase: Bool = false, _ replace: (Hit) -> String) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern, options: ignoreCase ? [.caseInsensitive] : []) else { return text }
        let ns = text as NSString
        var out = ""
        var last = 0
        for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            out += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            let groups: [String?] = (0..<m.numberOfRanges).map { i in
                let r = m.range(at: i)
                return r.location == NSNotFound ? nil : ns.substring(with: r)
            }
            out += replace(Hit(text: groups[0] ?? "", groups: groups, start: m.range.location, end: m.range.location + m.range.length))
            last = m.range.location + m.range.length
        }
        return out + ns.substring(from: last)
    }

    private static func isZh(_ c: Character) -> Bool {
        guard let v = c.unicodeScalars.first?.value else { return false }
        return v >= 0x4E00 && v <= 0x9FFF
    }

    private static func isLatin(_ c: Character) -> Bool { c.isASCII && c.isLetter }

    /// True when the nearest letter/CJK neighbour of text[s..<e] is Latin (English context).
    private static func enContext(_ text: String, _ s: Int, _ e: Int) -> Bool {
        let ns = text as NSString
        let left = ns.substring(to: s), right = ns.substring(from: e)
        let l = left.reversed().first { isZh($0) || isLatin($0) }
        let r = right.first { isZh($0) || isLatin($0) }
        for c in [l, r] {
            guard let c else { continue }
            if isLatin(c) { return true }
            if isZh(c) { return false }
        }
        return false
    }

    private static func yearEn(_ y: Int) -> String {
        if (2000...2009).contains(y) { return "two thousand" + (y == 2000 ? "" : " " + NumberWords.cardinalEn(y % 100)) }
        if (1000...2099).contains(y) {
            let r = y % 100
            let tail = r > 0 && r < 10 ? "oh " + NumberWords.cardinalEn(r) : (r > 0 ? NumberWords.cardinalEn(r) : "hundred")
            return NumberWords.cardinalEn(y / 100) + " " + tail
        }
        return NumberWords.cardinalEn(y)
    }

    private static func dd(_ digits: String, en: Bool) -> String { en ? NumberWords.digitsEn(digits) : NumberWords.digitsZh(digits) }

    // MARK: normalize

    static func normalize(_ input: String) -> String {
        if input.isEmpty { return input }
        var text = input

        // 1) email -> spelled
        text = sub(#"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"#, text) { m in
            var s = m.text.replacingOccurrences(of: "@", with: " at ").replacingOccurrences(of: ".", with: " dot ")
            s = sub(#"\d"#, s) { d in " " + NumberWords.digitsEn(d.text) + " " }
            return " " + collapse(s) + " "
        }

        // 1.5) time of day
        let t0 = text
        text = sub(#"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*([AaPp][.]?[Mm][.]?)?\b"#, t0) { m in
            let h = Int(m.groups[1]!)!, mm = m.groups[2]!, ap = m.groups[3]
            let mi = Int(mm)!
            let isAM = ap?.lowercased().contains("a") ?? false
            if enContext(t0, m.start, m.end) {
                let hh = NumberWords.cardinalEn(h)
                var t = mi == 0 ? "\(hh) o'clock" : (mi < 10 ? "\(hh) oh \(NumberWords.cardinalEn(mm))" : "\(hh) \(NumberWords.cardinalEn(mm))")
                if ap != nil { t += isAM ? " a m" : " p m" }
                return " " + t + " "
            }
            let pre = ap == nil ? "" : (isAM ? "上午" : "下午")
            let tail = mi == 0 ? "整" : (mi == 30 ? "半" : NumberWords.cardinalZh(mm) + "分")
            return " \(pre)\(NumberWords.cardinalZh(String(h)))點\(tail) "
        }

        // 2) date
        text = sub(#"(\d{4})\s*年\s*(?:(\d{1,2})\s*月)?\s*(?:(\d{1,2})\s*[日號])?"#, text) { m in
            var out = NumberWords.digitsZh(m.groups[1]!) + "年"
            if let mo = m.groups[2] { out += NumberWords.cardinalZh(mo) + "月" }
            if let d = m.groups[3] { out += NumberWords.cardinalZh(d) + "日" }
            return out
        }
        text = sub(#"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b"#, text) { m in
            var out = "\(m.groups[1]!) \(NumberWords.ordinalEn(m.groups[2]!))"
            if let y = m.groups[3], let yi = Int(y) { out += " " + yearEn(yi) }
            return out
        }

        // 3) temperature
        let t3 = text
        text = sub(#"(?:攝氏)?\s*([-－]?)(\d+)\s*(?:°\s*[Cc]|度[Cc]?|℃)"#, t3) { m in
            let neg = m.groups[1] == "-" || m.groups[1] == "－"
            let n = m.groups[2]!
            if enContext(t3, m.start, m.end) { return " \(neg ? "minus " : "")\(NumberWords.cardinalEn(n)) degrees Celsius " }
            return " 攝氏\(neg ? "零下" : "")\(NumberWords.cardinalZh(n))度 "
        }

        // 4) percent
        let t4 = text
        text = sub(#"(\d+)\s*[%％]"#, t4) { m in
            let n = m.groups[1]!
            return enContext(t4, m.start, m.end) ? " \(NumberWords.cardinalEn(n)) percent " : " 百分之\(NumberWords.cardinalZh(n)) "
        }

        // 5) price
        let t5 = text
        text = sub(#"(?:NT\$|US\$|USD|\$)\s*([\d,]+)(?:\.(\d{2}))?"#, t5) { m in
            let whole = m.groups[1]!.replacingOccurrences(of: ",", with: ""), cents = m.groups[2]
            let en = enContext(t5, m.start, m.end) || m.text.contains("USD") || m.text.contains("US$")
            var out: String
            if en {
                out = NumberWords.cardinalEn(whole) + " dollar" + (whole == "1" ? "" : "s")
                if let c = cents { out += " and " + NumberWords.cardinalEn(c) + " cent" + (c == "01" ? "" : "s") }
            } else {
                out = NumberWords.cardinalZh(whole) + "元"
                if let c = cents { out += NumberWords.cardinalZh(c) + "分" }
            }
            return " " + out + " "
        }
        text = sub(#"([\d,]+)\s*(元|塊錢|塊|台幣|新台幣)"#, text) { m in
            " " + NumberWords.cardinalZh(m.groups[1]!.replacingOccurrences(of: ",", with: "")) + (m.groups[2] ?? "元") + " "
        }

        // 6) phone / extension groups -> digit by digit
        let t6 = text
        text = sub(#"\(?\d{2,4}\)?(?:[-\s]\d{2,4}){1,4}"#, t6) { m in
            let en = enContext(t6, m.start, m.end)
            let trimmed = m.text.trimmingCharacters(in: CharacterSet(charactersIn: "()"))
            let groups = trimmed.components(separatedBy: CharacterSet(charactersIn: "-").union(.whitespacesAndNewlines))
                .filter { !$0.isEmpty }
                .map { $0.filter(\.isNumber) }
            return " " + groups.filter { !$0.isEmpty }.map { dd($0, en: en) }.joined(separator: " ") + " "
        }
        let t6b = text
        text = sub(#"(分機|內線|ext\.?|extension)\s*(\d{2,6})"#, t6b, ignoreCase: true) { m in
            m.groups[1]! + " " + dd(m.groups[2]!, en: enContext(t6b, m.start, m.end))
        }

        // 7) serial / order code
        text = sub(#"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]{2,}(?:-[A-Za-z0-9]+)*\b"#, text) { m in
            " " + m.text.filter { $0.isLetter || $0.isNumber }
                .map { $0.isNumber ? NumberWords.digitsEn(String($0)) : String($0).uppercased() }.joined(separator: " ") + " "
        }

        // 7.5) decimals
        let t75 = text
        text = sub(#"(?<![\d.])(\d+)\.(\d+)(?![\d.])"#, t75) { m in
            let whole = m.groups[1]!, frac = m.groups[2]!
            if enContext(t75, m.start, m.end) { return " \(NumberWords.cardinalEn(whole)) point \(NumberWords.digitsEn(frac)) " }
            return " \(NumberWords.cardinalZh(whole))點\(NumberWords.digitsZh(frac)) "
        }

        // 8) counts / remaining digit runs
        let t8 = text
        let t8ns = t8 as NSString
        text = sub(#"\d+"#, t8) { m in
            let d = m.text
            let en = enContext(t8, m.start, m.end)
            let after: Character? = m.end < t8ns.length ? Character(t8ns.substring(with: NSRange(location: m.end, length: 1))) : nil
            if d.count >= 5 { return dd(d, en: en) }
            if en { return d.count >= 4 ? NumberWords.digitsEn(d) : d }
            // Python quirk kept: at end of string `"" in _ADDR` is true, so no digit-by-digit there.
            if d.count >= 4, !(after.map { addr.contains($0) } ?? true) { return NumberWords.digitsZh(d) }
            return NumberWords.cardinalZh(d)
        }

        for (a, b) in [("，", ","), ("。", "."), ("？", "?"), ("！", "!")] { text = text.replacingOccurrences(of: a, with: b) }
        text = sub(#"[ \t]+"#, text) { _ in " " }
        text = sub(#"\s+([,.?!])"#, text) { $0.groups[1]! }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func collapse(_ s: String) -> String {
        s.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}
