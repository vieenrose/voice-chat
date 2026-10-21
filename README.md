# Jetson Nano 語音對話

完全在本機執行的語音對話，由 Nano（4 核心、約 4 GB 共用記憶體、CUDA 10.2 sm_53）以 WebUI 提供服務。

程式碼在 `jetson/`；以下路徑都以它為基準（在 Nano 上部署為 `~/voicechat-jetson`）。

| 階段 | 模型 | 執行裝置 |
|---|---|---|
| VAD | Silero | CPU |
| STT | X-ASR 480 ms chunk（sherpa-onnx，int8），由 VAD 觸發 | CPU |
| LLM | Bonsai-2 三元 4B（`run.sh` 的 `VC_LLM` 選擇 gguf，可換回 2B），`fwd35_serve`（PULSAR 引擎，`engine/make_serve.py`） | GPU |
| TTS | PrimeTTS v2.1（g2pW + VITS，onnxruntime） | CPU |
| 輪次偵測 | Smart Turn v3.2（`run.sh` 預設開啟，`VC_SMART_TURN=0` 關閉；調校中，見下方診斷） | CPU |

```bash
./run.sh        # 在 ~/voicechat-jetson 執行；服務位於 https://<host>:8443，token 在 server/lan_token
```

端點需要 token 才能存取。請勿公開綁定，也不要用 Funnel 對外開放；請使用 Tailscale。
首句切分需要 `jieba`（在 venv 中 `pip install jieba`）。

## WebUI 與展示

頁面（zh-TW）包含角色選單、即時逐字稿與最近一輪的延遲；右側是追蹤面板
（VAD／ASR／LLM／TTS／喇叭時間軸、各核心 CPU、GPU，以及最近一輪的時間花在哪裡）。

![WebUI，咖啡店店員情境（此角色已移除，截圖僅供版面參考）](docs/webui-roleplay.png)

此截圖是在 headless Chrome 中的真實對話，麥克風輸入為合成語音。這種語音會讓 X-ASR 把部分句子截短
（例如 `有什`）；真人說話則不會，逐字稿照 ASR 的輸出原樣呈現。

## 角色扮演

目前只保留一個角色：電信客服（`app/roles.py`）。咖啡店店員、面試官、台南導覽員對 2B 模型來說太難，
回覆常常跑出角色或重複，所以移除了。連線後伺服器會先用 TTS（不經 LLM）說出開場白，
系統提示中也會告訴模型這句話已經說過，並在開場白播放時先預填系統提示（`Llm.warm`），
第一輪回答就只需要處理使用者說的話。`run_turn` 中的回覆防護會在遇到空行、重複字元或重複詞句時
截斷回覆並取消生成：小模型否則會陷入循環（`點點點…`）或重複上一個問題。
`app/roles_check.py` 播放腳本對話（僅用 GPU），`app/role_test.py` 在執行中的伺服器上檢查協定。

## ASR 模型

使用官方 [X-ASR-zh-en](https://huggingface.co/GilgameshWind/X-ASR-zh-en) 的 480 ms chunk 串流模型（`run.sh` 的 `VC_ASR_DIR`，預設 `~/xasr-new/480`）。
官方只提供 fp32，encoder 與 joiner 用 `onnxruntime.quantization.quantize_dynamic`（MatMul、QUInt8）轉成 int8，
Nano 記憶體不足，所以在另一台機器上轉換。

串流 encoder 要多收到一個 chunk 的音訊才會解出最後幾個字，所以 `finish()` 在 VAD 判定說完後補 1.0 s 靜音
（`VC_ASR_FLUSH_S`）。0.6 s 會截掉句尾。在 40 句 `tw_cs` 上以 2 條執行緒、greedy 解碼量測：

| 模型 | 補靜音 | CER | RTF |
|---|---|---|---|
| 960 ms（舊的 `base`） | 1.0 s | 53.9% | 0.42 |
| **480 ms（採用）** | 1.0 s | **8.7%** | 0.66 |
| 160 ms | 1.0 s | 12.2% | 1.08 |
| 480 ms | 0.5 s | 21.6% | 0.53 |

RTF 含補的靜音，句子很短所以偏高。160 ms 在 2 條執行緒上超過即時，不適用。
代價是每句結尾多約 0.3 s 延遲（flush 從約 270 ms 變成約 580 ms）；0.5 s 與 1.0 s 之間的值尚未量測。

## CPU 配置

沒有綁定核心，也沒有調整優先權：每個元件分配固定的執行緒數，其餘交由 Linux 在 4 個核心間排程。

| 元件 | 執行緒 | 設定 |
|---|---|---|
| VAD | 1 | `app/vad.py` 中的 `intra_op_num_threads` |
| ASR | 2 | `VC_ASR_THREADS` |
| TTS（g2pW 與 VITS session） | 3 | `TTS_THREADS` |
| Smart Turn | 2 | `VC_SMART_TURN_THREADS` |
| LLM 引擎 | 1 個驅動執行緒，約 0.25 核心 | 受 GPU 限制 |

各階段大多依序執行（ASR、LLM、TTS）。真正重疊的只有 LLM 解碼時的 TTS，設定值來自
`app/sched_check.py`（背景有 LLM 解碼時，每句 TTS 的合成時間）：

| TTS 執行緒 | 單獨執行 | LLM 解碼同時 |
|---|---|---|
| 4 | 1.83 s | 2.36 s（+29%） |
| 3（採用） | 1.84 s | 2.02 s（+10%） |
| 2 | 2.53 s | 2.66 s（+5%） |

第 4 條執行緒單獨使用沒有加速，還會與引擎的驅動執行緒和網路迴圈競爭。把 TTS 綁到核心 1-3、引擎綁到
核心 0，只多快 0.02 s，所以不採用。ASR 用 2 條執行緒已是最佳（480 ms 模型的 flush 約 580 ms；3 條約 560 ms，4 條反而較慢 700 ms）。
插話（使用者在助理說話時開口，ASR 與 TTS 同時運作）尚未量測。

## 診斷

- `/trace`（也嵌入在對話頁面中）：各階段時間區段、各核心 CPU、GPU 負載與 RAM。
  LLM 區段的標籤來自引擎每次請求的統計行（`fwd35: … prefill … | decode … tok/s`）：
  - `prefill N reused`：沿用引擎狀態，只預填新增的 token，跳過前 N 個；`prefill full`：從頭預填。
    歷史視窗重新開始或上一輪被截斷時會出現 `full`，這是預期行為。若每一輪都是 `full`，代表 `prepare`
    產生的 token 與實際生成的不一致（例如回覆文字被改寫過），請檢查 `app/llm.py` 的 `prepare()`。
  - `decode N tok X tok/s`：本輪解碼的 token 數與速度；`(cut)` 表示被插話打斷。
  - 首個 token 那一段的文字標籤會顯示上述預填方式（對應 `/trace` 的「文字 → LLM 首個 token」列）。
  統計行只在引擎輸出時才會到達；若遲到超過 0.5 秒，標籤會省略，不影響語音輸出。
- 插話後（`cut_current`）會在背景重新預填歷史前綴（`prepare`），下一輪才能接續而不是從頭預填。
  這段在背景執行的錯誤會以 `prepare failed` 記錄在伺服器日誌中。
- `server/rec/`：最近 4 次對話的麥克風音訊（`.wav`）與事件記錄（`.log`：
  `speech_start`／`speech_end` 位移、最後的部分文字與最終文字，以及 Smart Turn 的 `turn_check p=…`）。
  檔案只留在裝置上，不納入版本控制。
- Smart Turn 調校（尚未定案）：每次 `speech_end` 會在記錄中寫入 `turn_check p=…`，p ≥ 0.5 視為回合結束。
  目前的觀察：離線重跑 25 個記錄片段時，24 個 p ≥ 0.5，唯一的例外是 p = 0.317 的一次完整句子；
  「就是我打開網頁的時候，那個就是沒有辦法」（p = 0.548）是唯一明顯未完成的片段，剛好在閾值之上。
  即時記錄中，「我檢查過了那個網絡線沒插」（句子在「沒插」後截斷，「沒插好」才完整）得到 p = 0.974，沒有被保留。
  結論：預設閾值 0.5 幾乎不會保留截斷的句子。需要更多在句中停頓的樣本才能選定閾值，同時估計 CPU 成本。
  離線分析的腳本未納入版本控制。
- 檢查腳本（需要 GPU 的請先停止伺服器）：`app/bench.py`、`app/text_check.py`、
  `app/llm_cont_check.py`、`app/tts_cut_check.py`、`app/tts_stream_check.py`、`app/sched_check.py`、
  `server/bargein_test.py`。
