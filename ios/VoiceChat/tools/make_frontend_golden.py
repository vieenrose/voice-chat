"""Reference output of the PrimeTTS python frontend, used to check the Swift port.

Run in a dir holding G2PWModel/ (config, txt lists, g2pw.onnx) with PrimeTTS scripts on PYTHONPATH:
  PYTHONPATH=<PrimeTTS>/scripts python make_frontend_golden.py out.json
"""
import json
import sys

import frontend_bopomofo as fb
import text_norm
from g2pw.dataset import TextDataset

SENTENCES = [
    "您好，請問有什麼可以幫您？",
    "台灣的首都是台北，人口大約兩百三十萬人。",
    "請撥分機 1234，或打 02-2345-6789。",
    "今天的氣溫是 28°C，降雨機率 70%。",
    "會議在 3:15 PM 開始，地點是 5 樓會議室。",
    "這件外套 NT$1,299，現在打八折。",
    "請寄信到 john.smith@example.com 謝謝。",
    "我想用 Python 寫一個 voice assistant。",
    "他行走在銀行旁邊，重新開始工作，長大以後覺得很快樂。",
    "今天是 2024年3月15日，星期五。",
    "My name is Luigi and I live in Taipei.",
    "Hello! 歡迎來到 Prime R&D，今天有什麼需要？",
    "那個東西的序號是 AB12345，請記下來。",
    "了解，還有其他問題嗎？沒有的話我就先掛電話了。",
    "圓周率大約是 3.14159，約等於三點一四。",
    "I have 250 apples, and the price is $49.99.",
    "你知道 Anthropic 這家公司嗎？",
    "約翰、瑪麗和彼得去了高雄的六合夜市。",
    "和合本聖經裡說：太初有道，道與神同在。",
    "Cafeteria is open from 9:30 AM to 5 PM.",
]

fb._lazy()
conv = fb._g2pw
conv.num_workers = 0  # no DataLoader worker processes
cases = []
for s in SENTENCES:
    norm = text_norm.normalize(s)
    # per-polyphone query features, as G2PWConverter builds them
    texts, qids, sids, partial = conv._prepare_data([norm])
    queries = []
    if texts:
        ds = TextDataset(conv.tokenizer, conv.labels, conv.char2phonemes, conv.chars, texts, qids,
                         use_mask=conv.config.use_mask, use_char_phoneme=conv.config.use_char_phoneme,
                         window_size=conv.config.window_size, for_train=False)
        for i in range(len(ds)):
            item = ds[i]
            queries.append({"query": qids[i], "input_ids": item["input_ids"].tolist(),
                            "position_id": item["position_id"], "char_id": item["char_id"],
                            "mask_ids": [k for k, v in enumerate(item["phoneme_mask"]) if v]})
    bopo = conv([norm])[0]
    ids = fb.text_to_ids(s)
    cases.append({"text": s, "norm": norm, "queries": queries, "bopomofo": bopo, **ids})

json.dump({"cases": cases, "an2cn": {str(n): __import__("cn2an").an2cn(n, "low") for n in
          [0, 1, 2, 9, 10, 11, 12, 19, 20, 21, 99, 100, 101, 110, 111, 120, 999, 1000, 1001, 1010, 1100, 2024, 9999, 10000, 10001, 12345, 100000, 123456, 1000000, 100000000]}},
          open(sys.argv[1], "w"), ensure_ascii=False, indent=0)
print(len(cases), "cases")
