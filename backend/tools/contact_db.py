"""
Contact DB — 100 zh-TW contacts with fuzzy phonetic matching for ASR/spelling errors.

Handles:
- Exact match → immediate
- Pinyin phonetic fuzzy (pypinyin + rapidfuzz/difflib) for ASR errors: 王/汪 (wang), 陳/程 (chen/cheng), 黃/王 (huang/wang with tone confusion), etc.
- All in zh-TW, responses in zh-TW.

DB: 100 real Taiwanese names (common surnames 林,陳,黃,張,李,王,吳,劉,蔡,楊...) + 4-digit extensions 1001-1099, depts.
"""
import re
from typing import List, Dict, Optional, Tuple

# 100 zh-TW contacts — common surnames + given names, extensions 1001-1100, depts
CONTACTS: List[Dict] = [
    {"name": "陳志明", "ext": "1001", "dept": "研發部", "pinyin": "chen zhi ming"},
    {"name": "林美玲", "ext": "1002", "dept": "行銷部", "pinyin": "lin mei ling"},
    {"name": "黃小華", "ext": "1003", "dept": "人資部", "pinyin": "huang xiao hua"},
    {"name": "張建國", "ext": "1004", "dept": "財務部", "pinyin": "zhang jian guo"},
    {"name": "李淑芬", "ext": "1005", "dept": "業務部", "pinyin": "li shu fen"},
    {"name": "王小明", "ext": "1006", "dept": "研發部", "pinyin": "wang xiao ming"},
    {"name": "吳宗憲", "ext": "1007", "dept": "行銷部", "pinyin": "wu zong xian"},
    {"name": "劉德華", "ext": "1008", "dept": "客服部", "pinyin": "liu de hua"},
    {"name": "蔡依林", "ext": "1009", "dept": "設計部", "pinyin": "cai yi lin"},
    {"name": "楊宗緯", "ext": "1010", "dept": "研發部", "pinyin": "yang zong wei"},
    {"name": "陳怡君", "ext": "1011", "dept": "行銷部", "pinyin": "chen yi jun"},
    {"name": "林志玲", "ext": "1012", "dept": "客服部", "pinyin": "lin zhi ling"},
    {"name": "黃志偉", "ext": "1013", "dept": "研發部", "pinyin": "huang zhi wei"},
    {"name": "張淑華", "ext": "1014", "dept": "人資部", "pinyin": "zhang shu hua"},
    {"name": "李建宏", "ext": "1015", "dept": "業務部", "pinyin": "li jian hong"},
    {"name": "王美華", "ext": "1016", "dept": "財務部", "pinyin": "wang mei hua"},
    {"name": "吳美玲", "ext": "1017", "dept": "設計部", "pinyin": "wu mei ling"},
    {"name": "劉志中", "ext": "1018", "dept": "研發部", "pinyin": "liu zhi zhong"},
    {"name": "蔡小芬", "ext": "1019", "dept": "行銷部", "pinyin": "cai xiao fen"},
    {"name": "楊美惠", "ext": "1020", "dept": "客服部", "pinyin": "yang mei hui"},
    {"name": "陳建宏", "ext": "1021", "dept": "業務部", "pinyin": "chen jian hong"},
    {"name": "林建宏", "ext": "1022", "dept": "研發部", "pinyin": "lin jian hong"},
    {"name": "黃美玲", "ext": "1023", "dept": "行銷部", "pinyin": "huang mei ling"},
    {"name": "張志強", "ext": "1024", "dept": "研發部", "pinyin": "zhang zhi qiang"},
    {"name": "李美華", "ext": "1025", "dept": "財務部", "pinyin": "li mei hua"},
    {"name": "王建國", "ext": "1026", "dept": "業務部", "pinyin": "wang jian guo"},
    {"name": "吳建國", "ext": "1027", "dept": "客服部", "pinyin": "wu jian guo"},
    {"name": "劉美玲", "ext": "1028", "dept": "設計部", "pinyin": "liu mei ling"},
    {"name": "蔡志明", "ext": "1029", "dept": "研發部", "pinyin": "cai zhi ming"},
    {"name": "楊志明", "ext": "1030", "dept": "行銷部", "pinyin": "yang zhi ming"},
    {"name": "陳美華", "ext": "1031", "dept": "財務部", "pinyin": "chen mei hua"},
    {"name": "林淑芬", "ext": "1032", "dept": "人資部", "pinyin": "lin shu fen"},
    {"name": "黃淑芬", "ext": "1033", "dept": "業務部", "pinyin": "huang shu fen"},
    {"name": "張美玲", "ext": "1034", "dept": "行銷部", "pinyin": "zhang mei ling"},
    {"name": "李志明", "ext": "1035", "dept": "研發部", "pinyin": "li zhi ming"},
    {"name": "王淑芬", "ext": "1036", "dept": "客服部", "pinyin": "wang shu fen"},
    {"name": "吳淑芬", "ext": "1037", "dept": "人資部", "pinyin": "wu shu fen"},
    {"name": "劉淑芬", "ext": "1038", "dept": "行銷部", "pinyin": "liu shu fen"},
    {"name": "蔡淑華", "ext": "1039", "dept": "業務部", "pinyin": "cai shu hua"},
    {"name": "楊淑芬", "ext": "1040", "dept": "財務部", "pinyin": "yang shu fen"},
    {"name": "陳小華", "ext": "1041", "dept": "研發部", "pinyin": "chen xiao hua"},
    {"name": "林小華", "ext": "1042", "dept": "行銷部", "pinyin": "lin xiao hua"},
    {"name": "黃小明", "ext": "1043", "dept": "客服部", "pinyin": "huang xiao ming"},
    {"name": "張小明", "ext": "1044", "dept": "設計部", "pinyin": "zhang xiao ming"},
    {"name": "李小明", "ext": "1045", "dept": "研發部", "pinyin": "li xiao ming"},
    {"name": "王志偉", "ext": "1046", "dept": "行銷部", "pinyin": "wang zhi wei"},
    {"name": "吳志偉", "ext": "1047", "dept": "客服部", "pinyin": "wu zhi wei"},
    {"name": "劉志偉", "ext": "1048", "dept": "研發部", "pinyin": "liu zhi wei"},
    {"name": "蔡志偉", "ext": "1049", "dept": "行銷部", "pinyin": "cai zhi wei"},
    {"name": "楊志偉", "ext": "1050", "dept": "財務部", "pinyin": "yang zhi wei"},
    {"name": "陳志偉", "ext": "1051", "dept": "業務部", "pinyin": "chen zhi wei"},
    {"name": "林志偉", "ext": "1052", "dept": "研發部", "pinyin": "lin zhi wei"},
    {"name": "黃志明", "ext": "1053", "dept": "行銷部", "pinyin": "huang zhi ming"},
    {"name": "張志明", "ext": "1054", "dept": "客服部", "pinyin": "zhang zhi ming"},
    {"name": "李志偉", "ext": "1055", "dept": "設計部", "pinyin": "li zhi wei"},
    {"name": "王志明", "ext": "1056", "dept": "研發部", "pinyin": "wang zhi ming"},
    {"name": "吳志明", "ext": "1057", "dept": "行銷部", "pinyin": "wu zhi ming"},
    {"name": "劉志明", "ext": "1058", "dept": "客服部", "pinyin": "liu zhi ming"},
    {"name": "蔡志中", "ext": "1059", "dept": "研發部", "pinyin": "cai zhi zhong"},
    {"name": "楊建國", "ext": "1060", "dept": "業務部", "pinyin": "yang jian guo"},
    {"name": "陳淑華", "ext": "1061", "dept": "財務部", "pinyin": "chen shu hua"},
    {"name": "林淑華", "ext": "1062", "dept": "人資部", "pinyin": "lin shu hua"},
    {"name": "黃淑華", "ext": "1063", "dept": "行銷部", "pinyin": "huang shu hua"},
    {"name": "張淑芬", "ext": "1064", "dept": "業務部", "pinyin": "zhang shu fen"},
    {"name": "李淑華", "ext": "1065", "dept": "財務部", "pinyin": "li shu hua"},
    {"name": "王淑華", "ext": "1066", "dept": "客服部", "pinyin": "wang shu hua"},
    {"name": "吳淑華", "ext": "1067", "dept": "人資部", "pinyin": "wu shu hua"},
    {"name": "劉淑華", "ext": "1068", "dept": "行銷部", "pinyin": "liu shu hua"},
    {"name": "蔡美玲", "ext": "1069", "dept": "設計部", "pinyin": "cai mei ling"},
    {"name": "楊美玲", "ext": "1070", "dept": "研發部", "pinyin": "yang mei ling"},
    {"name": "陳美玲", "ext": "1071", "dept": "行銷部", "pinyin": "chen mei ling"},
    {"name": "林美華", "ext": "1072", "dept": "財務部", "pinyin": "lin mei hua"},
    {"name": "黃美華", "ext": "1073", "dept": "業務部", "pinyin": "huang mei hua"},
    {"name": "張美華", "ext": "1074", "dept": "行銷部", "pinyin": "zhang mei hua"},
    {"name": "李美玲", "ext": "1075", "dept": "客服部", "pinyin": "li mei ling"},
    {"name": "王美玲", "ext": "1076", "dept": "設計部", "pinyin": "wang mei ling"},
    {"name": "吳美華", "ext": "1077", "dept": "研發部", "pinyin": "wu mei hua"},
    {"name": "劉美華", "ext": "1078", "dept": "行銷部", "pinyin": "liu mei hua"},
    {"name": "蔡美華", "ext": "1079", "dept": "客服部", "pinyin": "cai mei hua"},
    {"name": "楊美華", "ext": "1080", "dept": "財務部", "pinyin": "yang mei hua"},
    {"name": "陳建國", "ext": "1081", "dept": "業務部", "pinyin": "chen jian guo"},
    {"name": "林建國", "ext": "1082", "dept": "研發部", "pinyin": "lin jian guo"},
    {"name": "黃建國", "ext": "1083", "dept": "行銷部", "pinyin": "huang jian guo"},
    {"name": "張建宏", "ext": "1084", "dept": "客服部", "pinyin": "zhang jian hong"},
    {"name": "李建國", "ext": "1085", "dept": "設計部", "pinyin": "li jian guo"},
    {"name": "王建宏", "ext": "1086", "dept": "研發部", "pinyin": "wang jian hong"},
    {"name": "吳建宏", "ext": "1087", "dept": "行銷部", "pinyin": "wu jian hong"},
    {"name": "劉建宏", "ext": "1088", "dept": "客服部", "pinyin": "liu jian hong"},
    {"name": "蔡建宏", "ext": "1089", "dept": "研發部", "pinyin": "cai jian hong"},
    {"name": "楊建宏", "ext": "1090", "dept": "行銷部", "pinyin": "yang jian hong"},
    {"name": "陳志中", "ext": "1091", "dept": "財務部", "pinyin": "chen zhi zhong"},
    {"name": "林志中", "ext": "1092", "dept": "人資部", "pinyin": "lin zhi zhong"},
    {"name": "黃志中", "ext": "1093", "dept": "業務部", "pinyin": "huang zhi zhong"},
    {"name": "張志中", "ext": "1094", "dept": "行銷部", "pinyin": "zhang zhi zhong"},
    {"name": "李志中", "ext": "1095", "dept": "客服部", "pinyin": "li zhi zhong"},
    {"name": "王志中", "ext": "1096", "dept": "設計部", "pinyin": "wang zhi zhong"},
    {"name": "吳志中", "ext": "1097", "dept": "研發部", "pinyin": "wu zhi zhong"},
    {"name": "劉建國", "ext": "1098", "dept": "行銷部", "pinyin": "liu jian guo"},
    {"name": "蔡建國", "ext": "1099", "dept": "客服部", "pinyin": "cai jian guo"},
    {"name": "楊建中", "ext": "1100", "dept": "財務部", "pinyin": "yang jian zhong"},
]

# Build lookup dicts
_NAME_TO_CONTACT = {c["name"]: c for c in CONTACTS}
_PINYIN_TO_CONTACTS = {}  # pinyin -> list
for c in CONTACTS:
    _PINYIN_TO_CONTACTS.setdefault(c["pinyin"], []).append(c)

def _to_bopomofo(text: str) -> str:
    """Convert zh to bopomofo (zhuyin) using pypinyin Style.BOPOMOFO, fallback to char map. Bopomofo is zh-TW standard (ㄅㄆㄇㄈ), better than pinyin for TW ASR."""
    try:
        from pypinyin import pinyin, Style
        # BOPOMOFO includes tone marks, e.g. 王->ㄨㄤˊ, 小->ㄒㄧㄠˇ, 明->ㄇㄧㄥˊ
        pys = pinyin(text, style=Style.BOPOMOFO, errors='ignore')
        # pys is List[List[str]], flatten and join with space
        flat = [item[0] for sub in pys for item in [sub] if sub]
        return " ".join(flat)
    except:
        pass
    # Fallback: char -> bopomofo via CONTACTS bopomofo map (if pypinyin not available)
    char_map = {}
    for c in CONTACTS:
        # CONTACTS stores pinyin, not bopomofo, so fallback to pinyin as last resort
        for ch, py in zip(c["name"], c["pinyin"].split()):
            char_map[ch] = py
    pys = [char_map.get(ch, ch) for ch in text]
    return " ".join(pys)

def _similarity(a: str, b: str) -> float:
    """Phonetic similarity 0..1 using rapidfuzz if available, else difflib."""
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b) / 100.0
    except:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()

def find_contact(query: str, top_k: int = 3, threshold: float = 0.65) -> List[Tuple[Dict, float]]:
    """
    Fuzzy phonetic matching for zh-TW names with ASR/spelling errors.
    Returns list of (contact, score) sorted by score desc.
    Handles: 王小明 vs 王小名 (ming vs ming with typo), 陳/程 (chen vs cheng), 黃/王 pinyin confusion, etc.
    """
    query = query.strip()
    # Extract potential name from sentence: keep 2-4 consecutive zh chars that look like name
    # e.g., "請幫我查王小明的分機" -> "王小明"
    m = re.findall(r"[\u4e00-\u9fff]{2,4}", query)
    candidates_query = [query]  # original
    if m:
        # Prefer longest zh substring that could be name
        for cand in sorted(m, key=len, reverse=True):
            candidates_query.append(cand)
    # Also try removing common particles
    query_clean = re.sub(r"[請幫我查問一下的分機是幾號嗎呢啊？?]", "", query).strip()
    if query_clean and query_clean != query:
        candidates_query.append(query_clean)

    best = []
    seen = set()
    for q in candidates_query:
        q = q.strip()
        if not q or len(q) < 1:
            continue
        # 1. Exact match
        if q in _NAME_TO_CONTACT:
            return [(_NAME_TO_CONTACT[q], 1.0)]
        # 2. Substring exact
        for name, contact in _NAME_TO_CONTACT.items():
            if q == name or name == q:
                if name not in seen:
                    best.append((contact, 0.95))
                    seen.add(name)
        # 3. Bopomofo fuzzy (zh-TW zhuyin, handles ASR homophones better than pinyin)
        q_bpmf = _to_bopomofo(q)
        for contact in CONTACTS:
            name = contact["name"]
            if name in seen and any(s > 0.9 for _, s in best if _ == contact):
                continue
            # Cache bopomofo per contact (compute once)
            if "bopomofo" not in contact:
                contact["bopomofo"] = _to_bopomofo(name)
            c_bpmf = contact["bopomofo"]
            # Character-level Jaccard + pinyin similarity
            # Char similarity
            import difflib
            char_sim = difflib.SequenceMatcher(None, q, name).ratio()
            bpmf_sim = _similarity(q_bpmf, c_bpmf)
            # Weighted: bopomofo more important for TW ASR homophones (ㄨㄤˊ vs ㄨㄤˋ tone, ㄔㄣˊ vs ㄔㄥˊ)
            score = 0.35 * char_sim + 0.65 * bpmf_sim
            # Boost if pinyin edit distance 1 (tone confusion etc.)
            if score >= threshold and name not in seen:
                best.append((contact, score))
                seen.add(name)
    # Sort and dedup
    best = sorted(best, key=lambda x: x[1], reverse=True)
    # Deduplicate by name keep highest
    uniq = {}
    for c, s in best:
        if c["name"] not in uniq or s > uniq[c["name"]][1]:
            uniq[c["name"]] = (c, s)
    best = sorted(uniq.values(), key=lambda x: x[1], reverse=True)
    return best[:top_k]

def get_extension(query: str) -> Dict:
    """
    Main entry: query extension for a person (zh-TW, with spelling/ASR errors).
    Returns dict with found status, contact, or candidates.
    All messages in zh-TW.
    """
    query = query.strip()
    if not query:
        return {"found": False, "message": "請提供要查詢的姓名。", "candidates": []}
    matches = find_contact(query, top_k=3, threshold=0.6)
    if not matches:
        return {"found": False, "message": f"找不到「{query}」的相關聯絡人，請確認姓名是否正確。", "candidates": []}
    top, score = matches[0]
    if score >= 0.82:
        # High confidence — direct answer
        return {
            "found": True,
            "contact": top,
            "score": score,
            "message": f"{top['name']} 的分機是 {top['ext']}（{top['dept']}）。",
            "candidates": [{"name": c["name"], "ext": c["ext"], "dept": c["dept"], "score": s} for c, s in matches]
        }
    elif score >= 0.65 and len(matches) == 1:
        return {
            "found": True,
            "contact": top,
            "score": score,
            "message": f"您是指 {top['name']} 嗎？他的分機是 {top['ext']}（{top['dept']}）。",
            "candidates": [{"name": c["name"], "ext": c["ext"], "dept": c["dept"], "score": s} for c, s in matches]
        }
    else:
        # Multiple candidates
        cands = [{"name": c["name"], "ext": c["ext"], "dept": c["dept"], "score": round(s, 2)} for c, s in matches]
        names = "、".join([c["name"] for c, _ in matches])
        return {
            "found": False,
            "message": f"找到多位相似的聯絡人：{names}，請問您是指哪一位？",
            "candidates": cands
        }

# Quick test
if __name__ == "__main__":
    tests = ["王小明", "王小名", "王曉明", "汪小明", "陳志明", "程志明", "黃小華", "黃曉華", "請幫我查王小明的分機", "林美玲的分機", "張建國是幾號", "找不到的人"]
    for q in tests:
        res = get_extension(q)
        print(f"Q: {q:12} -> {res['message']} (found={res['found']}, score={res.get('score','')})")
