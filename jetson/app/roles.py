"""Role-play scenes: a system prompt, a TTS voice and an optional spoken opening line per role.
Only the telecom support role is kept: the 2B model could not hold the others (café, interview, guide).

The UI lists them (`roles` message) and the user picks one (`role` message); switching restarts the
conversation. The opening line is spoken straight through the TTS (no LLM call), and the system prompt
tells the model it was already said so the first answer continues from it.
Voices of primetts_v21_3voice: 0 Xinran (f), 1 Anchen (m), 2 Bowen (m).
"""
STYLE = ("請用台灣繁體中文，口語、簡短地說話，每次只說一到兩句話，只用一段話，不要重複自己說過的話。"
         "不要使用 Markdown、列表、星號、省略號或表情符號。")
IN_CHARACTER = ("你正在角色扮演，全程保持這個角色，絕對不要說自己是語音助理、人工智慧或語言模型。"
                "每次回答都要貼近情境，並盡量以一個簡短的問題結尾，讓對話繼續下去。")

ROLES = [
    {"id": "support", "name": "電信客服", "blurb": "你是來電客人，反映網路問題。", "voice": 0,
     "opening": "感謝來電，我是客服小林，請問有什麼可以幫您？",
     "system": "你是一家電信公司的客服人員小林，使用者是來電反映問題的客人。"
               "態度有禮貌、有耐心，會先確認問題，再一步一步引導對方排除狀況。" + STYLE + IN_CHARACTER},
]
BY_ID = {r["id"]: r for r in ROLES}


def system_prompt(role):
    """System prompt, with the already-spoken opening line so the model continues from it."""
    if not role["opening"]:
        return role["system"]
    return role["system"] + "對話一開始你已經說過：「%s」" % role["opening"]


def listing():
    return [{"id": r["id"], "name": r["name"], "blurb": r["blurb"]} for r in ROLES]
