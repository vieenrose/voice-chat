import Foundation

/// English word -> ARPAbet (with stress digits): CMUdict lookup, then g2p_en's seq2seq GRU for unknown words.
final class G2PEn {
    private var cmu: [String: [String]] = [:]
    private var net: Net?

    private static let graphemes = ["<pad>", "<unk>", "</s>"] + "abcdefghijklmnopqrstuvwxyz".map(String.init)
    private static let phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [
        "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2",
        "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2",
        "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1",
        "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2",
        "V", "W", "Y", "Z", "ZH"]

    init(cmudict: String, weights: Data) {
        for line in cmudict.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard parts.count > 1 else { continue }
            cmu[String(parts[0])] = parts.dropFirst().map(String.init)
        }
        net = Net(weights)
    }

    func pronounce(_ word: String) -> [String] {
        let w = word.lowercased()
        if let p = cmu[w] { return p }
        return net?.predict(w) ?? []
    }

    /// GRU seq2seq, hidden size 256, greedy decoding (port of G2p.predict).
    final class Net {
        let h = 256
        var enc_emb: [Float], enc_w_ih: [Float], enc_w_hh: [Float], enc_b_ih: [Float], enc_b_hh: [Float]
        var dec_emb: [Float], dec_w_ih: [Float], dec_w_hh: [Float], dec_b_ih: [Float], dec_b_hh: [Float]
        var fc_w: [Float], fc_b: [Float]

        init?(_ data: Data) {
            let all = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let sizes = [29 * 256, 768 * 256, 768 * 256, 768, 768, 74 * 256, 768 * 256, 768 * 256, 768, 768, 74 * 256, 74]
            guard all.count == sizes.reduce(0, +) else { return nil }
            var o = 0
            func take(_ n: Int) -> [Float] { defer { o += n }; return Array(all[o..<o + n]) }
            enc_emb = take(sizes[0]); enc_w_ih = take(sizes[1]); enc_w_hh = take(sizes[2]); enc_b_ih = take(sizes[3]); enc_b_hh = take(sizes[4])
            dec_emb = take(sizes[5]); dec_w_ih = take(sizes[6]); dec_w_hh = take(sizes[7]); dec_b_ih = take(sizes[8]); dec_b_hh = take(sizes[9])
            fc_w = take(sizes[10]); fc_b = take(sizes[11])
        }

        /// out[r] = sum_c w[r*cols + c] * x[c] + b[r]
        private func matvec(_ w: [Float], _ x: [Float], _ b: [Float], rows: Int) -> [Float] {
            let cols = x.count
            var out = b
            w.withUnsafeBufferPointer { wp in
                for r in 0..<rows {
                    var sum: Float = 0
                    let base = r * cols
                    for c in 0..<cols { sum += wp[base + c] * x[c] }
                    out[r] += sum
                }
            }
            return out
        }

        private func cell(_ x: [Float], _ hPrev: [Float], _ wih: [Float], _ whh: [Float], _ bih: [Float], _ bhh: [Float]) -> [Float] {
            let gi = matvec(wih, x, bih, rows: 3 * h), gh = matvec(whh, hPrev, bhh, rows: 3 * h)
            var next = [Float](repeating: 0, count: h)
            for i in 0..<h {
                let r = 1 / (1 + exp(-(gi[i] + gh[i])))
                let z = 1 / (1 + exp(-(gi[h + i] + gh[h + i])))
                let n = tanhf(gi[2 * h + i] + r * gh[2 * h + i])
                next[i] = (1 - z) * n + z * hPrev[i]
            }
            return next
        }

        func predict(_ word: String) -> [String] {
            let ids = word.map { G2PEn.graphemes.firstIndex(of: String($0)) ?? 1 } + [2]
            var hidden = [Float](repeating: 0, count: h)
            for id in ids { hidden = cell(Array(enc_emb[id * 256..<(id + 1) * 256]), hidden, enc_w_ih, enc_w_hh, enc_b_ih, enc_b_hh) }
            var token = 2
            var out: [String] = []
            for _ in 0..<20 {
                hidden = cell(Array(dec_emb[token * 256..<(token + 1) * 256]), hidden, dec_w_ih, dec_w_hh, dec_b_ih, dec_b_hh)
                let logits = matvec(fc_w, hidden, fc_b, rows: 74)
                var best = 0
                for k in 1..<74 where logits[k] > logits[best] { best = k }
                if best == 3 { break }
                out.append(G2PEn.phonemes[best])
                token = best
            }
            return out
        }
    }
}
