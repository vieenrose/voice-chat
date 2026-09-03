<script>
  import { onDestroy } from 'svelte'
  import { RealtimeClient } from './lib/realtime.js'
  import { MicCapture, Playback } from './lib/audio.js'
  import { toTW, toTWP, zhtwReady } from './lib/zhtw.js'
  import { serverUrl as buildUrl } from './lib/endpoints.js'

  const INSTRUCTIONS =
    '你是一個親切的語音助理。一律使用繁體中文（台灣用語）回答，' +
    '無論問題用什麼語言提出；只有專有名詞或無法翻譯的術語才保留英文。回答要口語、簡潔。'

  // Same rule as realtime.js: https implies a TLS terminator on 8443, because a
  // browser blocks ws:// from an https page and the pipeline speaks plain ws.
  const defaultUrl = () =>
    location.protocol === 'https:'
      ? `wss://${location.hostname}:8443/v1/realtime`
      : `ws://${location.hostname}:8765/v1/realtime`
  let url = $state(defaultUrl())
  let connected = $state(false)
  let connecting = $state(false)
  let listening = $state(false)
  let error = $state('')

  let turns = $state([])           // {role:'user'|'assistant', text, cancelled?}
  let userPartial = $state('')
  let userSpeaking = $state(false)
  let responding = $state(false)
  let speaking = $state(false)
  let lastStatus = $state('')

  let level = $state(0)            // mic peak, for the meter
  let vram = $state(null)          // {used_mib, total_mib, device} from the server
  let llmCfg = $state(null)        // {model, api_base, tools} -- read-only, for the card
  let lookup = $state(null)        // newest search_contacts result, for the panel

  const PIPELINE = $derived([
    ['VAD', 'Silero v5 + Smart Turn v3.2'],
    ['STT', '無（Gemma 4 原生語音輸入）'],
    ['LLM', llmCfg ? llmCfg.model : '無法取得（伺服器未啟動？）'],
    ['Agent', `自建工具迴圈 · 原生工具呼叫（${llmCfg?.tools?.length ?? '?'} 工具，上限 3 步）`],
    ['TTS', 'Qwen3-TTS 12Hz · GGML 24k'],
  ])
  let textInput = $state('')
  // Light by default. index.html carries class="light" so the pre-mount paint
  // matches and the page does not flash dark before Svelte takes over.
  let theme = $state('light')

  // Latency: from the user's speech ending (or text send) to first audio out.
  let tSpeechEnd = 0
  let firstAudioMs = $state(0)
  let audioSeconds = $state(0)

  let client = null
  let mic = null
  let play = null
  let chatEl = $state(null)

  // The Realtime server exposes GET /v1/vram (added in s2s/serve.py). Derive its URL
  // from the endpoint the user is already connected to, so it follows the TLS proxy.
  const vramUrl = () => buildUrl(url, '/v1/vram')

  const serverUrl = (path) => buildUrl(url, path)

  async function loadLlmCfg() {
    const target = serverUrl('/v1/llm-config')
    if (!target) return
    try {
      const r = await fetch(target, { cache: 'no-store' })
      llmCfg = r.ok ? await r.json() : null
    } catch {
      llmCfg = null
    }
  }

  // The pipeline card is informational, so it should not wait for the user to
  // press 連線: the model row read 尚未連線 while the server was perfectly
  // reachable. Fetched once on mount, and again on connect in case the endpoint
  // field was edited in between.
  $effect(() => {
    loadLlmCfg()
  })







  // The Realtime protocol has no server->client tool-result event (ServerEvent is a
  // closed union of OpenAI types), so the directory result is read from
  // GET /v1/tool-trace, the same HTTP surface as /v1/vram.
  async function pollLookup() {
    const target = serverUrl('/v1/tool-trace')
    if (!target) return
    try {
      const r = await fetch(target, { cache: 'no-store' })
      const { trace = [] } = await r.json()
      const last = [...trace].reverse().find((e) => e.name === 'search_contacts')
      lookup = last ? { ...last, matches: last.result?.matches ?? [] } : null
    } catch {
      /* server down or origin refuses the fetch; keep whatever is on screen */
    }
  }

  async function pollVram() {
    const target = vramUrl()
    if (!target) return
    try {
      const r = await fetch(target, { cache: 'no-store' })
      const d = await r.json()
      vram = d.available ? d : null
    } catch {
      vram = null   // server down, or an origin that refuses the fetch
    }
  }

  let vramTimer = null
  $effect(() => {
    // Poll only while connected: the readout is for watching the model load and run.
    if (connected && !vramTimer) {
      pollVram(); pollLookup()
      // The lookup panel is polled faster than VRAM: it should appear while the
      // assistant is still speaking the question about which department.
      vramTimer = setInterval(() => { pollVram(); pollLookup() }, 1000)
    } else if (!connected && vramTimer) {
      clearInterval(vramTimer)
      vramTimer = null
      vram = null
    }
  })

  function scroll() {
    queueMicrotask(() => { if (chatEl) chatEl.scrollTop = chatEl.scrollHeight })
  }

  function appendAssistant(text) {
    if (!text) return
    const last = turns[turns.length - 1]
    if (last && last.role === 'assistant' && !last.done) {
      // Convert the whole accumulated reply, not each chunk: twp substitutes
      // multi-character phrases, which a chunk boundary can split.
      last.raw = (last.raw || '') + text
      last.text = toTWP(last.raw)
      turns = turns
    } else {
      turns = [...turns, { role: 'assistant', raw: text, text: toTWP(text), done: false }]
    }
    scroll()
  }

  function sealAssistant(cancelled) {
    const last = turns[turns.length - 1]
    if (last && last.role === 'assistant') {
      last.done = true
      last.cancelled = cancelled
      turns = turns
    }
  }

  const handlers = {
    onSpeechStart() {
      userSpeaking = true
      // Barge-in: the server cancels the response, and the audio already scheduled
      // in the browser has to be dropped too or the reply keeps talking locally.
      if (play?.playing) {
        play.flush()
        speaking = false
        note('barge_in.local_flush', { secondsPlayed: +audioSeconds.toFixed(2) })
      }
    },
    onSpeechStop() {
      userSpeaking = false
      tSpeechEnd = performance.now()
      firstAudioMs = 0
      // A VAD-triggered turn gets no response.created -- the server only emits that
      // for an explicit response.create -- so end-of-speech is the signal that the
      // reply is being worked on. Without this the "thinking" state never showed
      // for the voice path, which is the main path.
      responding = true
      lastStatus = ''
    },
    // Cumulative: replace, never append. Converted for display only -- Paraformer
    // emits Simplified, and this is a zh-TW demo.
    onUserPartial(text) { userPartial = toTW(text); scroll() },
    onUserFinal(text) {
      userPartial = ''
      // Keep the raw STT output alongside the displayed form: the export is for
      // debugging, and a converted transcript would hide what STT actually heard.
      if (text.trim()) turns = [...turns, { role: 'user', text: toTW(text), raw: text }]
      scroll()
    },
    // The caption arrives as a partial and never completes: it is produced off the
    // pipeline by X-ASR purely for display, so no transcription.completed follows.
    // Promote it to a real turn when the assistant starts answering, or it would sit
    // as transient italic text and be overwritten by the next utterance.
    promotePartial() {
      const t = userPartial.trim()
      if (!t) return
      userPartial = ''
      turns = [...turns, { role: 'user', text: t, raw: t, caption: true }]
      scroll()
    },
    onResponseStart() { responding = true; lastStatus = '' },
    onAssistantText(t) {
      if (userPartial.trim()) handlers.promotePartial()
      appendAssistant(t)
    },
    onAudio(b64) {
      if (!firstAudioMs && tSpeechEnd) firstAudioMs = Math.round(performance.now() - tSpeechEnd)
      audioSeconds += play.push(b64)
      speaking = true
    },
    onResponseDone(status, reason) {
      responding = false
      lastStatus = status === 'cancelled' ? `已中斷（${reason || 'cancelled'}）` : ''
      sealAssistant(status === 'cancelled')
    },
    onError(msg) { error = msg; note('server.error', { message: msg }) },
    onClose() { connected = false; listening = false; speaking = false },
  }

  async function connect() {
    error = ''
    connecting = true
    try {
      play = new Playback({ sampleRate: 24000 })
      play.onEnded = () => { speaking = false }
      client = new RealtimeClient(handlers)
      await client.connect(url, INSTRUCTIONS)
      connected = true
      loadLlmCfg()
    } catch (e) {
      // The usual cause is not "not running" but "bound to loopback": the page is
      // reachable over Tailscale/LAN while the server only listens on 127.0.0.1.
      const remote = location.hostname !== 'localhost' && location.hostname !== '127.0.0.1'
      error =
        `連線失敗：${e.message}。` +
        (remote
          ? '從其他機器連線時，https 需要 TLS 轉送：tailscale serve --bg --https 8443 http://127.0.0.1:8765'
          : '伺服器是否在執行？python3 -m s2s.serve --mode realtime --ws_host 0.0.0.0')
    } finally {
      connecting = false
    }
  }

  function disconnect() {
    // Snapshot everything the export needs BEFORE dropping the objects that hold
    // it. Without this the log exported after a session was empty -- protocol: []
    // with null rates and zero counters -- which is exactly when it gets clicked.
    lastSession = snapshotSession()
    stopMic()
    client?.close(); client = null
    play?.close(); play = null
    connected = false; speaking = false; responding = false
  }

  async function startMic() {
    error = ''
    try {
      mic = new MicCapture({
        sampleRate: 16000,
        onChunk: (b64) => client?.appendAudio(b64),
        onLevel: (p) => { level = p },
      })
      await mic.start()
      listening = true
      note('mic.started', { rate: mic.sampleRate, secure: window.isSecureContext })
    } catch (e) {
      error = `麥克風無法啟動：${e.message}`
      note('mic.error', { message: e.message, name: e.name, secure: window.isSecureContext })
    }
  }

  function stopMic() {
    // Hand the VAD its silence before the audio stops arriving, or an utterance
    // that was still open when the user hit 停止收音 never closes and never gets
    // answered. Only worth doing while the socket is still up -- on disconnect the
    // turn is being abandoned anyway.
    if (connected) client?.flushSilence()
    mic?.stop(); mic = null
    listening = false; level = 0
  }

  function sendText() {
    const t = textInput.trim()
    if (!t || !connected) return
    turns = [...turns, { role: 'user', text: t }]
    tSpeechEnd = performance.now()
    firstAudioMs = 0
    client.sendText(t)
    textInput = ''
    scroll()
  }

  function interrupt() {
    note('interrupt.button')
    client?.cancel()
    play?.flush()
    speaking = false
  }

  // --- debug export -------------------------------------------------------
  // A single .json holding the protocol trace plus the client-side state, so a
  // report of "it said the wrong thing" can be read back without guessing what
  // the browser saw. Audio payloads are elided in the client's own log.
  let clientEvents = $state([])   // things only the UI knows: flushes, mic errors
  let lastSession = null          // survives disconnect, so the export still has it

  function note(what, detail = {}) {
    clientEvents.push({
      t: +performance.now().toFixed(1),
      wall: new Date().toISOString(),
      what,
      ...detail,
    })
    if (clientEvents.length > 500) clientEvents.shift()
  }

  /** The parts of the log that live on objects disconnect() throws away. */
  function snapshotSession() {
    return {
      audio: {
        playbackRate: play?.sampleRate ?? null,
        captureRate: mic?.sampleRate ?? null,
        micFrames: client?.micFrames ?? 0,
        micBase64Bytes: client?.micBytes ?? 0,
        audioBase64BytesIn: client?.audioBytesIn ?? 0,
      },
      protocol: client?.log ? [...client.log] : [],
    }
  }

  function buildLog() {
    // Live objects when connected; the disconnect snapshot afterwards.
    const snap = client ? snapshotSession() : (lastSession ?? snapshotSession())
    return {
      exported: new Date().toISOString(),
      page: { url: location.href, protocol: location.protocol, hostname: location.hostname },
      endpoint: url,
      agent: navigator.userAgent,
      state: { connected, listening, responding, speaking, lastStatus, error },
      audio: {
        firstAudioMs: firstAudioMs || null,
        secondsPlayed: +audioSeconds.toFixed(2),
        ...snap.audio,
      },
      transcript: turns.map((t) => ({
        role: t.role,
        text: t.text,
        ...(t.raw && t.raw !== t.text ? { rawStt: t.raw } : {}),
        cancelled: !!t.cancelled,
        done: !!t.done,
      })),
      clientEvents,
      // Mic frames are counted above rather than listed; everything else is here.
      protocol: snap.protocol,
    }
  }

  function exportLog() {
    const blob = new Blob([JSON.stringify(buildLog(), null, 2)], { type: 'application/json' })
    const href = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = href
    a.download = `voice-chat-log-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
    document.body.appendChild(a)
    a.click()
    a.remove()
    // Revoked on a later tick: revoking immediately can cancel the download in
    // some browsers before it has read the blob.
    setTimeout(() => URL.revokeObjectURL(href), 10000)
  }

  function toggleTheme() {
    theme = theme === 'dark' ? 'light' : 'dark'
    document.body.classList.toggle('light', theme === 'light')
  }

  // Keep the DOM in step with the initial state, not just with later toggles.
  $effect(() => {
    document.body.classList.toggle('light', theme === 'light')
  })

  // The converter arrives asynchronously, so any turn captured before it landed
  // is still Simplified. Re-render those from the raw STT text once it is live.
  zhtwReady.then(() => {
    let changed = false
    for (const t of turns) {
      if (!t.raw) continue
      const conv = t.role === 'assistant' ? toTWP(t.raw) : toTW(t.raw)
      if (conv !== t.text) { t.text = conv; changed = true }
    }
    if (changed) turns = turns
  })

  onDestroy(disconnect)
</script>

<div class="wrap">
  <header class="header">
    <div class="brand">
      <div class="logo">🎙️</div>
      <div style="min-width:0">
        <div class="title">語音分機查詢 · Extension Lookup by Voice</div>
        <div class="subtitle">說出同事姓名，問到分機 · HuggingFace speech-to-speech · 全本地 · 繁體中文</div>
      </div>
    </div>
    <div class="header-actions">
      <span class="pill {connected ? 'ok' : ''}">{connected ? '● Live' : '○ Offline'}</span>
      {#if listening}<span class="pill ok">🎤 收音中</span>{/if}
      {#if vram}
        <span class="pill" title="{vram.device} — 整張卡的用量，含 llama-server"
              style="font-variant-numeric:tabular-nums">
          VRAM {(vram.used_mib / 1024).toFixed(1)}/{(vram.total_mib / 1024).toFixed(1)} GB
        </span>
      {/if}
      <button class="ghost" style="padding:6px 10px; font-size:12px" onclick={exportLog}
              title="下載這次工作階段的完整記錄（JSON），用於回報問題">⬇ 記錄</button>
      <button class="ghost" style="padding:6px 10px; font-size:12px" onclick={toggleTheme}
              aria-label={theme === 'dark' ? '切換為亮色主題' : '切換為深色主題'}>
        {theme === 'dark' ? '☀️' : '🌙'}
      </button>
    </div>
  </header>

  <div class="grid">
    <div class="stack controls-col">
      <div class="card">
        <h3>連線</h3>
        <div class="controls">
          {#if !connected}
            <button class="primary" onclick={connect} disabled={connecting}>
              {connecting ? '連線中…' : '連線'}
            </button>
          {:else}
            <button class="ghost" onclick={disconnect}>中斷連線</button>
          {/if}
          {#if connected && !listening}
            <button class="primary" onclick={startMic}>🎤 開始說話</button>
          {:else if listening}
            <button class="danger" onclick={stopMic}>停止收音</button>
          {/if}
          {#if responding || speaking}
            <button class="ghost" onclick={interrupt}>⏹ 中斷回覆</button>
          {/if}
        </div>

        <label for="ws-url" class="subtle" style="display:block;margin-bottom:4px">Realtime 端點</label>
        <input id="ws-url" class="input" bind:value={url} disabled={connected} />

        <div class="meter" aria-hidden="true">
          <div class="meter-fill" style="width:{Math.min(100, level * 180)}%"></div>
        </div>

        {#if userSpeaking}
          <div class="status-banner status-speaking"><span class="dot">●</span> 偵測到說話——助理會停止並聆聽</div>
        {:else if speaking}
          <div class="status-banner status-speaking"><span class="dot">●</span> 助理正在說話——直接開口即可打斷</div>
        {:else if responding}
          <div class="status-banner status-speaking"><span class="dot">●</span> 思考中…（可能含網路搜尋）</div>
        {/if}
        {#if lastStatus}
          <div class="status-banner status-cancel">{lastStatus}</div>
        {/if}
        {#if error}
          <div class="status-banner status-err">⚠️ {error}</div>
        {/if}

        <div class="lat-grid">
          <div class="lat"><b>{firstAudioMs || '—'}</b><span>首次出聲 ms</span></div>
          <div class="lat"><b>{audioSeconds.toFixed(1)}</b><span>已播秒數</span></div>
        </div>
      </div>

      {#if lookup}
        <div class="card">
          <h3>分機查詢
            <span class="subtle" style="font-weight:400">
              {lookup.arguments?.query ?? ''}{lookup.arguments?.department ? ` · ${lookup.arguments.department}` : ''}
            </span>
          </h3>
          {#if lookup.matches.length === 0}
            <div class="subtle" style="font-size:12px">查無此人</div>
          {:else}
            {#if lookup.matches.length > 1}
              <div class="subtle" style="font-size:11px; margin-bottom:6px">
                {lookup.matches.length} 位同名，待確認部門
              </div>
            {/if}
            <table class="dir">
              <thead><tr><th>姓名</th><th>部門</th><th>職稱</th><th>分機</th></tr></thead>
              <tbody>
                {#each lookup.matches as m}
                  <tr><td>{m.name}</td><td>{m.dept}</td><td>{m.title}</td><td class="ext">{m.ext}</td></tr>
                {/each}
              </tbody>
            </table>
          {/if}
        </div>
      {/if}

      <div class="card">
        <h3>管線</h3>
        <div style="font-size:12px; line-height:1.7; opacity:0.85">
          {#each PIPELINE as [k, v]}
            <div><b>{k}</b> {v}</div>
          {/each}
          <div class="subtle" style="margin-top:6px">
            打斷由伺服器的 CancelScope 處理；瀏覽器同時清掉已排程的音訊。
          </div>
        </div>
      </div>
    </div>

    <div class="stack chat-col">
      <div class="card">
        <h3>對話</h3>
        <div class="chat" bind:this={chatEl}>
          {#each turns as t}
            <div class="bubble {t.role}" class:cancelled={t.cancelled}
                 title={t.caption ? 'X-ASR 字幕（僅供顯示，模型直接聽語音）' : null}>{t.text}</div>
          {/each}
          {#if userPartial}
            <div class="bubble user partial">{userPartial}</div>
          {/if}
          {#if !turns.length && !userPartial}
            <div class="subtle" style="margin:auto; text-align:center; font-size:12px">
              按「連線」，再按「開始說話」。<br />也可以直接打字。
            </div>
          {/if}
        </div>
        <div class="composer">
          <input class="input" placeholder="輸入訊息…" bind:value={textInput}
                 onkeydown={(e) => e.key === 'Enter' && sendText()} disabled={!connected} />
          <button class="primary" onclick={sendText} disabled={!connected || !textInput.trim()}>送出</button>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  :global(*){box-sizing:border-box}
  :global(body){margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; background:#0a0a0f; color:#e8e8ef; line-height:1.5}
  :global(body.light){background:#f8f9fc; color:#1a1a24}
  .wrap{max-width:1120px; margin:0 auto; padding:24px 16px 32px}
  .header{position:sticky; top:0; z-index:10; backdrop-filter:blur(12px); background:rgba(10,10,15,0.85); border:1px solid #1e1e28; border-radius:16px; padding:14px 16px; display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:16px}
  :global(body.light) .header{background:rgba(255,255,255,0.9); border-color:#e8eaf0}
  .brand{display:flex; gap:12px; align-items:center; min-width:0}
  .logo{width:36px; height:36px; border-radius:10px; background:#7c5cff; display:grid; place-items:center; font-size:18px; flex-shrink:0}
  .title{font-size:15px; font-weight:700; letter-spacing:-0.01em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .subtitle{font-size:11px; opacity:0.55; margin-top:2px}
  table.dir{width:100%; border-collapse:collapse; font-size:12px}
  table.dir th{text-align:left; font-weight:500; opacity:0.5; padding:2px 6px 4px 0}
  table.dir td{padding:4px 6px 4px 0; border-top:1px solid rgba(128,128,128,0.18)}
  table.dir td.ext{font-variant-numeric:tabular-nums; font-weight:600}
  .header-actions{display:flex; gap:8px; align-items:center; flex-wrap:wrap}
  .pill{font-size:11px; padding:5px 9px; border-radius:999px; border:1px solid #232332; background:#14141c}
  .pill.ok{border-color:#7c5cff; color:#b8a6ff}
  :global(body.light) .pill{background:#fff; border-color:#e3e4ee; color:#555}
  .grid{display:grid; grid-template-columns:380px 1fr; gap:16px}
  @media(max-width:900px){
    .grid{grid-template-columns:1fr}
    .header{flex-direction:column; align-items:stretch}
    .chat-col{order:-1}
  }
  .stack{display:flex; flex-direction:column; gap:16px}
  .card{background:#14141c; border:1px solid #1e1e28; border-radius:14px; padding:14px}
  :global(body.light) .card{background:#fff; border-color:#e8eaf0; box-shadow:0 1px 3px rgba(0,0,0,0.06)}
  .card h3{font-size:11px; font-weight:600; letter-spacing:0.07em; text-transform:uppercase; opacity:0.6; margin:0 0 10px 0}
  .controls{display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px}
  button{appearance:none; border:0; padding:9px 14px; border-radius:999px; font-weight:600; font-size:13px; cursor:pointer; transition:all .12s}
  button:disabled{opacity:0.4; cursor:not-allowed}
  .primary{background:#7c5cff; color:#fff}
  .primary:hover:not(:disabled){transform:translateY(-1px); box-shadow:0 4px 12px rgba(124,92,255,0.3)}
  .ghost{background:#1e1e28; color:#d8d8e6; border:1px solid #232332}
  .ghost:hover{background:#252538}
  :global(body.light) .ghost{background:#f1f2f6; color:#2a2a3a; border-color:#e3e4ee}
  .danger{background:#ff3b5e; color:#fff}
  .input{width:100%; padding:9px 11px; border-radius:8px; border:1px solid #232332; background:#0f0f14; color:inherit; font-size:13px; font-family:inherit}
  :global(body.light) .input{background:#f8f9fc; border-color:#e3e4ee}
  .meter{height:6px; border-radius:999px; background:#0f0f14; border:1px solid #1e1e28; margin:10px 0; overflow:hidden}
  :global(body.light) .meter{background:#f0f1f8; border-color:#eef0f6}
  .meter-fill{height:100%; background:#7c5cff; transition:width .06s linear}
  .lat-grid{display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-top:10px}
  .lat{padding:10px 6px; border-radius:10px; background:#0f0f14; border:1px solid #1e1e28; text-align:center}
  .lat b{font-size:15px; display:block; font-variant-numeric:tabular-nums}
  .lat span{font-size:10px; opacity:0.55; letter-spacing:0.06em}
  :global(body.light) .lat{background:#f8f9fc; border-color:#eef0f6}
  .status-banner{display:flex; gap:8px; align-items:center; font-size:12px; padding:8px 10px; border-radius:8px; margin:8px 0}
  .status-speaking{background:#14101f; border:1px solid #2a2242; color:#c7b8ff}
  :global(body.light) .status-speaking{background:#f3f0ff; border-color:#ded4ff; color:#5a3fc0}
  .status-cancel{background:#1a1510; border:1px solid #3a2c1a; color:#ffd9a3}
  .status-err{background:#1a0f14; border:1px solid #3a1a1a; color:#ffb3b3}
  .status-banner .dot{animation:pulse 1.2s infinite; color:#7c5cff}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.25}}
  .chat{height:460px; overflow:auto; display:flex; flex-direction:column; gap:8px; padding:4px; scroll-behavior:smooth}
  .bubble{max-width:82%; padding:10px 12px; border-radius:12px; font-size:13px; line-height:1.6; word-break:break-word; white-space:pre-wrap}
  .bubble.user{align-self:flex-end; background:#7c5cff; color:#fff; border-bottom-right-radius:4px}
  .bubble.assistant{align-self:flex-start; background:#1e1e28; border:1px solid #232332; border-bottom-left-radius:4px}
  :global(body.light) .bubble.assistant{background:#f1f2f6; border-color:#e3e4ee}
  .bubble.partial{opacity:0.55; font-style:italic}
  .bubble.cancelled{opacity:0.6}
  .bubble.cancelled::after{content:' ⏹'; opacity:0.7}
  .composer{display:flex; gap:8px; margin-top:10px}
  .composer .input{flex:1 1 auto; min-width:0}
  .composer button{flex:0 0 auto}
  .subtle{opacity:0.55}
</style>
