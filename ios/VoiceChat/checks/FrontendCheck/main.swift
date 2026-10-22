import Foundation

// Compares the Swift PrimeTTS frontend with reference output from the Python frontend.
// Usage: FrontendCheck <golden.json> <Resources/tts dir>

let args = CommandLine.arguments
let golden = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[1]))) as! [String: Any]
let dir = URL(fileURLWithPath: args[2])
var failures = 0
func fail(_ msg: String) { failures += 1; print("FAIL:", msg) }

// numbers
let an2cn = golden["an2cn"] as! [String: String]
for (k, v) in an2cn.sorted(by: { Int($0.key)! < Int($1.key)! }) where NumberWords.cardinalZh(Int(k)!) != v {
    fail("an2cn \(k): got \(NumberWords.cardinalZh(Int(k)!)) want \(v)")
}
let en = golden["en"] as! [String: [String: String]]
for (k, v) in en["card"]! where NumberWords.cardinalEn(Int(k)!) != v { fail("card \(k): got \(NumberWords.cardinalEn(Int(k)!)) want \(v)") }
for (k, v) in en["ord"]! where NumberWords.ordinalEn(Int(k)!) != v { fail("ord \(k): got \(NumberWords.ordinalEn(Int(k)!)) want \(v)") }

let resData = try Data(contentsOf: dir.appendingPathComponent("g2pw.json"))
let resources = try JSONDecoder().decode(G2PW.Resources.self, from: resData)
let vocab = try String(contentsOf: dir.appendingPathComponent("vocab.txt"), encoding: .utf8)
let g2pw = G2PW(resources: resources, vocab: vocab)
let g2pen = G2PEn(cmudict: try String(contentsOf: dir.appendingPathComponent("cmudict.txt"), encoding: .utf8),
                  weights: try Data(contentsOf: dir.appendingPathComponent("g2pen.bin")))
let symbolsJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("symbol_table.json"))) as! [String: Any]
let symbols = symbolsJSON["symbols"] as! [String]
let labelIndex = Dictionary(uniqueKeysWithValues: resources.labels.enumerated().map { ($1, $0) })

for c in golden["cases"] as! [[String: Any]] {
    let text = c["text"] as! String
    let norm = TextNorm.normalize(text)
    if norm != (c["norm"] as! String) { fail("norm [\(text)]\n   got  \(norm)\n   want \(c["norm"] as! String)") }

    // tokenization features of every polyphone query
    let chars = norm.unicodeScalars.map { String($0) }
    let wantQueries = c["queries"] as! [[String: Any]]
    var gotQueries: [G2PW.Query] = []
    for (i, ch) in chars.enumerated() where resources.char2ph[ch] != nil {
        if let q = g2pw.makeQuery(chars: chars, index: i) { gotQueries.append(q) }
    }
    if gotQueries.count != wantQueries.count { fail("query count [\(text)]: \(gotQueries.count) vs \(wantQueries.count)") }
    for (g, w) in zip(gotQueries, wantQueries) {
        let ids = (w["input_ids"] as! [Int]).map(Int64.init)
        if g.inputIds != ids || g.position != Int64(w["position_id"] as! Int) || g.charId != Int64(w["char_id"] as! Int)
            || Set(g.allowed) != Set(w["mask_ids"] as! [Int]) {
            fail("query \(g.index) [\(text)] ids \(g.inputIds) vs \(ids)")
        }
    }

    // stub scorer: answers with the reference reading so only the surrounding logic is under test
    let wantBopomofo = c["bopomofo"] as! [Any]
    var cursor = 0
    var order: [Int] = []
    for (i, ch) in chars.enumerated() where resources.char2ph[ch] != nil { order.append(i) }
    let frontend = PrimeFrontend(symbols: symbols, g2pw: g2pw, g2pen: g2pen) { ids, _, _, _ in
        defer { cursor += ids.count }
        return (0..<ids.count).map { k in
            var row = [Float](repeating: 0, count: resources.labels.count)
            if let label = wantBopomofo[order[cursor + k]] as? String, let idx = labelIndex[label] { row[idx] = 1 }
            return row
        }
    }
    let out = try frontend.convert(text)
    let wantIds = (c["phone_ids"] as! [Int]).map(Int64.init)
    if out.phoneIds != wantIds {
        fail("phones [\(text)]\n   got  \(out.phones.joined(separator: " "))\n   want \((c["phones"] as! [String]).joined(separator: " "))")
    } else if out.tones != (c["tone_ids"] as! [Int]).map(Int64.init) || out.langs != (c["lang_ids"] as! [Int]).map(Int64.init) {
        fail("tones/langs [\(text)]")
    }
}
print(failures == 0 ? "ALL OK" : "\(failures) failures")
exit(failures == 0 ? 0 : 1)
