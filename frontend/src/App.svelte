<script>
  import { onMount, onDestroy } from 'svelte';
  import { marked } from 'marked';
  import DOMPurify from 'dompurify';
  marked.setOptions({ breaks: true, gfm: true });
  function renderMarkdown(text){
    try{ const html = marked.parse(text || ''); return DOMPurify.sanitize(html); }catch(e){ return (text||'').replace(/\n/g,'<br/>'); }
  }
  let s2tConverter = null;
  let toTraditional = (text) => text;
  async function initOpenCC(){
    try{
      const OpenCC = await import('opencc-js');
      s2tConverter = OpenCC.Converter({ from: 'cn', to: 'tw' });
      toTraditional = (text) => {
        try { return text && /[\u4e00-\u9fff]/.test(text) ? s2tConverter(text) : text; } catch(e) { return text; }
      };
    }catch(e){ console.warn('OpenCC not loaded', e); }
  }
  let connected = $state(false);
  let connecting = $state(false);
  let listening = $state(false);
  let speaking = $state(false);
  let useWorklet = $state(true);
  let vadEnabled = $state(true);
  let sttPartial = $state('');
  let sttFinal = $state('');
  let llmStreaming = $state('');
  let ttsText = $state('');
  let chatHistory = $state([]);
  let latency = $state({stt_ms:0, llm_ttft_ms:0, tts_ttfb_ms:0, e2e_ms:0});
  let rssMb = $state(0);
  let audioLevel = $state(0);
  let mode = $state('mock');
  let ws = null;
  let audioCtx = null;
  let workletNode = null;
  let mediaStream = null;
  let analyser = null;
  let jitterQueue = [];
  let nextPlayTime = 0;
  let wsUrl = $derived((location.protocol==='https:' ? `wss://${location.host}/ws/chat?session_id=${'demo-'+Math.random().toString(36).slice(2,7)}` : `ws://${location.hostname}:8000/ws/chat?session_id=${'demo-'+Math.random().toString(36).slice(2,7)}`));
  let statsInterval = null;
  let canvasEl;
  let animId;
  function log(msg, obj) { console.log('[UI]', msg, obj||''); }
  async function fetchHealth(){
    try{
      const healthUrl = location.protocol==='https:' ? '/health' : `http://${location.hostname}:8000/health`;
      const r = await fetch(healthUrl);
      const j = await r.json();
      rssMb = j.rss_mb;
      mode = j.mock ? 'mock' : 'real';
      searxngOk = j.searxng?.ok || false;
      return j;
    }catch(e){ rssMb=0; return null; }
  }
  async function testSearch(q){ sendText(q); }
  async function connect(){
    if (connected || connecting) return;
    connecting = true;
    let url = wsUrl;
    const qp = new URLSearchParams(location.search);
    if (qp.get('ws')) url = qp.get('ws');
    ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { connected=true; connecting=false; log('WS open', url); chatHistory = [...chatHistory, {role:'system', text:`Connected ${url} — mock=${mode}`}]; };
    ws.onclose = () => { connected=false; connecting=false; listening=false; log('WS closed'); };
    ws.onerror = (e) => { log('WS error', e); connecting=false; };
    ws.onmessage = async (ev) => {
      try{ const data = JSON.parse(ev.data); handleServerMessage(data); }catch(e){}
    };
  }
  function disconnect(){ if(ws){ ws.close(); ws=null; } connected=false; stopMic(); }
  let toolStatus = $state('');
  let audioError = $state('');
  let lastSearchResults = $state([]);
  let searxngOk = $state(false);
  function handleServerMessage(msg){
    switch(msg.type){
      case 'stt_partial': sttPartial = toTraditional(msg.text); break;
      case 'stt_final': sttFinal = toTraditional(msg.text); sttPartial = ''; chatHistory = [...chatHistory, {role:'user', text: toTraditional(msg.text)}]; llmStreaming = ''; ttsText = ''; toolStatus = ''; break;
      case 'llm_token': llmStreaming = toTraditional(msg.text_so_far); break;
      case 'tool_call': {
          const tn = msg.name || 'tool'; const q = msg.query || msg.arguments?.query || msg.arguments?.timezone || '';
          if(tn === 'web_search'){ toolStatus = `🔍 web_search("${q}")`; chatHistory = [...chatHistory, {role:'tool', text: `🔍 Searching "${q}"…`}]; }
          else { toolStatus = `🕐 ${tn}()`; chatHistory = [...chatHistory, {role:'tool', text: `🕐 ${tn}()`}]; }
        } break;
      case 'tool_result': {
          const src = msg.source || msg.result?.source || 'searxng'; const latency = msg.latency_ms || 0; const count = msg.result?.results?.length || 3;
          toolStatus = `✓ ${count} results via ${src} · ${latency}ms`; lastSearchResults = msg.result?.results || [];
          let preview = (msg.formatted||'').slice(0,200).replace(/\n/g,' '); chatHistory = [...chatHistory, {role:'tool', text: `✓ ${src} · ${latency}ms — ${preview}`}]; } break;
      case 'tts_chunk': ttsText = toTraditional(msg.text); if(msg.pcm){ try{ const pcm = base64ToInt16(msg.pcm); queueAudio(pcm, msg.sampleRate || 16000); }catch(e){ audioError = String(e).slice(0,120); } } updateAssistantStreaming(); break;
      case 'tts_start': preRollStarted = false; preRollQueue.length = 0; speaking = true; break;
      case 'tts_end': flushPreRoll(); speaking = false; finalizeAssistant(); toolStatus = ''; break;
      case 'latency': latency = msg; break;
    }
  }
  function updateAssistantStreaming(){
    const displayText = toTraditional(llmStreaming);
    if(displayText){
      if(chatHistory.length && chatHistory[chatHistory.length-1].role==='assistant' && chatHistory[chatHistory.length-1].streaming){
        chatHistory[chatHistory.length-1].text = displayText;
        chatHistory = [...chatHistory];
      } else { chatHistory = [...chatHistory, {role:'assistant', text: displayText, streaming:true}]; }
    }
  }
  function finalizeAssistant(){
    const displayText = toTraditional(llmStreaming);
    if(chatHistory.length && chatHistory[chatHistory.length-1].streaming){ chatHistory[chatHistory.length-1].streaming = false; chatHistory[chatHistory.length-1].text = displayText; chatHistory = [...chatHistory]; }
    else if(displayText){ chatHistory = [...chatHistory, {role:'assistant', text: displayText}]; }
    llmStreaming='';
  }
  function base64ToInt16(b64){ const bin = atob(b64); const bytes = new Uint8Array(bin.length); for(let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i); return new Int16Array(bytes.buffer); }
  function floatTo16BitPCM(float32){ const out = new Int16Array(float32.length); for(let i=0;i<float32.length;i++){ let s = Math.max(-1, Math.min(1, float32[i])); out[i] = s < 0 ? s*0x8000 : s*0x7FFF; } return out; }
  async function startMic(){
    micError=''; if(listening) return; if(!connected){ await connect(); await new Promise(r=>setTimeout(r,400)); }
    if(!window.isSecureContext){ micError = `Mic blocked: not secure context.`; chatHistory=[...chatHistory,{role:'system',text: micError}]; }
    try{
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate:16000});
      if(audioCtx.state === 'suspended') await audioCtx.resume();
      mediaStream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, sampleRate:16000, echoCancellation:true, noiseSuppression:true, autoGainControl:true}});
    }catch(e){ micError = `${e.name}: ${e.message}`; chatHistory=[...chatHistory,{role:'system',text: micError}]; return; }
    const src = audioCtx.createMediaStreamSource(mediaStream);
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 256; src.connect(analyser);
    if(useWorklet){
      try{
        await audioCtx.audioWorklet.addModule('/audio-processor.js');
        workletNode = new AudioWorkletNode(audioCtx, 'capture-processor');
        src.connect(workletNode);
        workletNode.port.onmessage = (e)=>{ if(e.data.type==='chunk'){ const pcm16 = floatTo16BitPCM(e.data.pcm); let rms=0; for(let i=0;i<e.data.pcm.length;i++) rms+=e.data.pcm[i]*e.data.pcm[i]; audioLevel=Math.min(1,Math.sqrt(rms/e.data.pcm.length)*8); sendPCM(pcm16); } };
        listening = true;
      }catch(e){ useWorklet=false; startScriptProcessor(src); }
    } else { startScriptProcessor(src); }
    drawWave();
  }
  function startScriptProcessor(src){
    const proc = audioCtx.createScriptProcessor(512,1,1); let buffer = new Float32Array(0);
    proc.onaudioprocess = (e)=>{ const input = e.inputBuffer.getChannelData(0); const combined = new Float32Array(buffer.length + input.length); combined.set(buffer); combined.set(input, buffer.length); buffer = combined; while(buffer.length >= 320){ const chunk = buffer.slice(0,320); buffer = buffer.slice(320); const pcm16 = floatTo16BitPCM(chunk); let rms=0; for(let i=0;i<chunk.length;i++) rms+=chunk[i]*chunk[i]; audioLevel=Math.min(1,Math.sqrt(rms/chunk.length)*6); sendPCM(pcm16); } };
    src.connect(proc); proc.connect(audioCtx.destination); const gain = audioCtx.createGain(); gain.gain.value=0; proc.connect(gain); workletNode = proc; listening=true;
  }
  function stopMic(){ listening=false; if(animId) cancelAnimationFrame(animId); if(workletNode){ try{workletNode.disconnect();}catch(e){} workletNode=null; } if(mediaStream){ mediaStream.getTracks().forEach(t=>t.stop()); mediaStream=null; } if(audioCtx){ try{audioCtx.close();}catch(e){} audioCtx=null; } audioLevel=0; if(ws && ws.readyState===1) ws.send(JSON.stringify({type:'stop'})); }
  function sendPCM(pcm16){ if(!ws || ws.readyState!==1) return; const header = new Uint8Array(1); header[0]=0x01; const out = new Uint8Array(1 + pcm16.byteLength); out.set(header,0); out.set(new Uint8Array(pcm16.buffer),1); try{ ws.send(out); }catch(e){ const b64 = btoa(String.fromCharCode(...new Uint8Array(pcm16.buffer))); ws.send(JSON.stringify({type:'audio_chunk', pcm:b64, sampleRate:16000})); } }
  function sendText(text){ if(!ws || ws.readyState!==1){ connect(); setTimeout(()=>sendText(text),500); return; } if(speaking) bargeIn(); ws.send(JSON.stringify({type:'text_input', text})); chatHistory=[...chatHistory, {role:'user', text}]; llmStreaming=''; }
  let activeSources = [];
  function bargeIn(){ if(ws && ws.readyState===1) ws.send(JSON.stringify({type:'barge_in'})); jitterQueue=[]; preRollQueue.length=0; preRollStarted=false; nextPlayTime=0; speaking=false;
    for(const s of activeSources){ try{ s.stop(); }catch(e){} try{ s.disconnect(); }catch(e){} }
    activeSources=[];
    if(playCtx){ try{playCtx.close();}catch(e){} playCtx=null; }
    // Clear any queued TTS chunks
    ttsText='';
  }
  let playCtx = null;
  function ensurePlayCtx(sampleRate){
    if(!playCtx){ try{ playCtx = new (window.AudioContext||window.webkitAudioContext)(); }catch(e){ playCtx = new (window.AudioContext||window.webkitAudioContext)(); } nextPlayTime = playCtx.currentTime;
      const _resume = () => { if(playCtx && playCtx.state==='suspended') playCtx.resume(); }; document.addEventListener('pointerdown', _resume, {once:true}); window.addEventListener('touchend', _resume, {once:true});
    } else if(playCtx.state==='suspended'){ try{ playCtx.resume(); }catch(e){} } return playCtx;
  }
  function resampleLinear(input, inRate, outRate){ if(inRate===outRate) return input; const ratio = inRate / outRate; const outLen = Math.round(input.length / ratio); const out = new Float32Array(outLen); for(let i=0;i<outLen;i++){ const pos=i*ratio; const idx=Math.floor(pos); const frac=pos-idx; out[i]=(input[idx]||0)+( (input[idx+1]||0)-(input[idx]||0))*frac; } return out; }
  const PRE_ROLL_SEC = 0.6; const preRollQueue = []; let preRollStarted = false;
  function queueAudio(pcm16, sampleRate){
    const ctx = ensurePlayCtx(sampleRate); const float = new Float32Array(pcm16.length); for(let i=0;i<pcm16.length;i++) float[i]=pcm16[i]/32768; const resampled = resampleLinear(float, sampleRate, ctx.sampleRate);
    const buf = ctx.createBuffer(1, resampled.length, ctx.sampleRate); buf.getChannelData(0).set(resampled); const src = ctx.createBufferSource(); src.buffer=buf; src.connect(ctx.destination); const dur=buf.duration;
    activeSources.push(src);
    src.onended = ()=>{ activeSources = activeSources.filter(s=>s!==src); if(ctx.currentTime >= nextPlayTime - 0.05) setTimeout(()=>{ if(ctx.currentTime >= nextPlayTime && activeSources.length===0) speaking=false; }, 200); };
    if(!preRollStarted){ preRollQueue.push({buf, src, dur}); if(preRollQueue.reduce((a,c)=>a+c.dur,0) >= PRE_ROLL_SEC) flushPreRoll(); return; }
    if(nextPlayTime < ctx.currentTime || !isFinite(nextPlayTime)) nextPlayTime = ctx.currentTime + 0.08; src.start(nextPlayTime); nextPlayTime += dur; speaking=true;
  }
  function flushPreRoll(){ const ctx=ensurePlayCtx(); if(!preRollQueue.length || preRollStarted) return; preRollStarted=true; if(nextPlayTime < ctx.currentTime || !isFinite(nextPlayTime)) nextPlayTime = ctx.currentTime + 0.08; for(const c of preRollQueue){ try{ c.src.start(nextPlayTime); }catch(e){} nextPlayTime+=c.dur; } preRollQueue.length=0; speaking=true; }
  function drawWave(){
    if(!canvasEl || !analyser) return; const ctx = canvasEl.getContext('2d'); const data = new Uint8Array(analyser.frequencyBinCount);
    const draw = ()=>{ animId=requestAnimationFrame(draw); analyser.getByteFrequencyData(data); ctx.clearRect(0,0,canvasEl.width,canvasEl.height); const w=canvasEl.width,h=canvasEl.height; const barW=w/data.length*2.2; let x=0; for(let i=0;i<data.length;i+=2){ const v=data[i]/255; const bh=v*h*0.85; ctx.fillStyle=listening?`hsla(265,90%,65%,${0.45+v*0.5})`:`hsla(220,8%,45%,0.5)`; ctx.fillRect(x,h-bh,barW,bh); x+=barW+1; if(x>=w) break; } ctx.fillStyle=listening?'#7c5cff':'#3a3a44'; ctx.fillRect(0,h-3,w*audioLevel,3); }; draw();
  }
  onMount(()=>{ fetchHealth(); statsInterval=setInterval(fetchHealth,3000); initOpenCC(); });
  onDestroy(()=>{ disconnect(); if(statsInterval) clearInterval(statsInterval); if(animId) cancelAnimationFrame(animId); });
  let micError = $state('');
  let textInput = $state('');
  let theme = $state('dark');
  $effect(()=>{ try{ theme=localStorage.getItem('vc-theme')||'dark'; }catch(e){} document.body.classList.toggle('light', theme==='light'); });
  function toggleTheme(){ theme=theme==='dark'?'light':'dark'; try{localStorage.setItem('vc-theme',theme);}catch(e){} document.body.classList.toggle('light', theme==='light'); }
  function clearChat(){ chatHistory=[]; }
</script>

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
  .header-actions{display:flex; gap:8px; align-items:center; flex-wrap:wrap}
  .pill{font-size:11px; padding:5px 9px; border-radius:999px; border:1px solid #232332; background:#14141c; display:inline-flex; gap:5px; align-items:center}
  .pill.ok{border-color:#7c5cff; color:#b8a6ff}
  :global(body.light) .pill{background:#fff; border-color:#e3e4ee; color:#555}
  :global(body.light) .pill.ok{border-color:#7c5cff; color:#6a4beb}
  .grid{display:grid; grid-template-columns: 380px 1fr; gap:16px}
  @media(max-width:900px){ .grid{grid-template-columns:1fr} .header{flex-direction:column; align-items:stretch} }
  .card{background:#14141c; border:1px solid #1e1e28; border-radius:14px; padding:14px}
  :global(body.light) .card{background:#fff; border-color:#e8eaf0; box-shadow:0 1px 3px rgba(0,0,0,0.06)}
  .card h3{font-size:11px; font-weight:600; letter-spacing:0.07em; text-transform:uppercase; opacity:0.6; margin:0 0 10px 0}
  .controls{display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px}
  button{appearance:none; border:0; padding:9px 14px; border-radius:999px; font-weight:600; font-size:13px; cursor:pointer; transition:all .12s}
  button:disabled{opacity:0.4; cursor:not-allowed}
  .primary{background:#7c5cff; color:#fff}
  .primary:hover{transform:translateY(-1px); box-shadow:0 4px 12px rgba(124,92,255,0.3)}
  .ghost{background:#1e1e28; color:#d8d8e6; border:1px solid #232332}
  .ghost:hover{background:#252538}
  :global(body.light) .ghost{background:#f1f2f6; color:#2a2a3a; border-color:#e3e4ee}
  .danger{background:#ff3b5e; color:#fff}
  .lat-grid{display:grid; grid-template-columns: repeat(4,1fr); gap:8px}
  .lat{padding:10px 6px; border-radius:10px; background:#0f0f14; border:1px solid #1e1e28; text-align:center}
  .lat b{font-size:15px; display:block; font-variant-numeric:tabular-nums}
  .lat span{font-size:10px; opacity:0.55; text-transform:uppercase; letter-spacing:0.06em}
  :global(body.light) .lat{background:#f8f9fc; border-color:#eef0f6}
  .wave{width:100%; height:64px; border-radius:10px; background:#0a0a0f; border:1px solid #1e1e28; margin:10px 0; display:block}
  :global(body.light) .wave{background:#f0f1f8; border-color:#eef0f6}
  .chat{height:420px; overflow:auto; display:flex; flex-direction:column; gap:8px; padding:4px; scroll-behavior:smooth}
  .bubble{max-width:82%; padding:10px 12px; border-radius:12px; font-size:13px; line-height:1.5; word-break:break-word}
  .bubble.user{align-self:flex-end; background:#7c5cff; color:#fff; border-bottom-right-radius:4px}
  .bubble.assistant{align-self:flex-start; background:#1e1e28; border:1px solid #232332}
  .bubble.system{align-self:center; background:transparent; border:1px dashed #2a2a3a; font-size:11px; opacity:0.7}
  .bubble.tool{align-self:center; background:#0e1a14; border:1px solid #1e3326; color:#8fd9b0; font-size:11px; max-width:90%; border-radius:8px; padding:7px 10px}
  .bubble.markdown :global(p){margin:6px 0}
  .bubble.markdown :global(strong){color:#fff; font-weight:700}
  :global(body.light) .bubble.markdown :global(strong){color:#1a1a24}
  .bubble.markdown :global(ul), .bubble.markdown :global(ol){margin:6px 0 6px 18px; padding:0}
  .bubble.markdown :global(li){margin:4px 0}
  .bubble.markdown :global(a){color:#8ea6ff; text-decoration:underline}
  .bubble.markdown :global(h1), .bubble.markdown :global(h2), .bubble.markdown :global(h3){margin:8px 0 4px; font-size:13px}
  :global(body.light) .bubble.assistant{background:#f1f2f8; border-color:#e3e4ee}
  :global(body.light) .bubble.system{background:#f8f9fc; border-color:#e3e4ee}
  :global(body.light) .bubble.tool{background:#eef8f0; border-color:#c8e8d0; color:#2a6b4a}
  .meta{font-size:11px; opacity:0.5; margin-top:6px}
  .row{display:flex; gap:8px; margin-top:10px}
  .input{flex:1; background:#0f0f14; border:1px solid #232332; color:#e8e8ef; padding:10px 12px; border-radius:999px; outline:none; font-size:13px}
  .input:focus{border-color:#7c5cff}
  :global(body.light) .input{background:#fff; border-color:#e3e4ee; color:#1a1a24}
  .toggles{display:flex; gap:10px; font-size:11px; opacity:0.8; margin:8px 0; align-items:center; flex-wrap:wrap}
  .kpi{display:flex; gap:6px; font-size:11px; opacity:0.65; flex-wrap:wrap}
  .kpi div{background:#0f0f14; padding:5px 8px; border-radius:999px; border:1px solid #1e1e28}
  :global(body.light) .kpi div{background:#f8f9fc; border-color:#eef0f6}
  .tool-bar{display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; align-items:center}
  .tool-chip{font-size:11px; padding:6px 10px; border-radius:999px; background:#1a1a1c; border:1px solid #2a2a3a; color:#ccc; cursor:pointer}
  .tool-chip:hover{border-color:#7c5cff; color:#b8a6ff}
  :global(body.light) .tool-chip{background:#fff; border-color:#e3e4ee; color:#555}
  .stack{display:flex; flex-direction:column; gap:12px}
  .subtle{font-size:11px; opacity:0.55; line-height:1.5}
  @keyframes pulse{0%{opacity:0.4}50%{opacity:1}100%{opacity:0.4}}
</style>

<div class="wrap">
  <header class="header">
    <div class="brand">
      <div class="logo">🎙️</div>
      <div style="min-width:0">
        <div class="title">Voice Chat</div>
        <div class="subtitle">X-ASR-int8 · Qwen3.5-0.8B Q8_0 · Granite-97M Q8 (CUDA) · PrimeTTS v2 16k · Qwen-Agent</div>
      </div>
    </div>
    <div class="header-actions">
      <span class="pill {connected ? 'ok' : ''}">{connected ? '● Live' : '○ Offline'} · {mode.toUpperCase()}</span>
      <span class="pill" style="font-variant-numeric:tabular-nums">{rssMb} MB</span>
      <button class="ghost" style="padding:6px 10px; font-size:12px" onclick={toggleTheme}>{theme==='dark' ? '☀️' : '🌙'}</button>
    </div>
  </header>

  <div class="grid">
    <div class="stack">
      <div class="card">
        <h3>Realtime</h3>
        <div class="controls">
          {#if !connected}
            <button class="primary" onclick={connect} disabled={connecting}>{connecting ? 'Connecting…' : 'Connect'}</button>
          {:else}
            <button class="ghost" onclick={disconnect}>Disconnect</button>
          {/if}
          {#if !listening}
            <button class="primary" onclick={startMic} disabled={!connected}>🎤 Listen</button>
          {:else}
            <button class="danger" onclick={stopMic}>Stop</button>
            <button class="ghost" onclick={bargeIn}>Barge-in</button>
          {/if}
          <button class="ghost" onclick={()=>{vadEnabled=!vadEnabled}}>{vadEnabled ? 'VAD' : 'VAD off'}</button>
        </div>
        {#if micError}
          <div style="margin:8px 0; padding:10px; background:#1a0f14; border:1px solid #3a1a1a; border-radius:8px; font-size:12px; color:#ffb3b3">{micError}</div>
        {/if}
        <canvas bind:this={canvasEl} class="wave" width="640" height="64"></canvas>
        <div class="lat-grid">
          <div class="lat"><b>{latency.stt_ms||0}</b><span>STT</span></div>
          <div class="lat"><b>{latency.llm_ttft_ms||0}</b><span>LLM</span></div>
          <div class="lat"><b>{latency.tts_ttfb_ms||0}</b><span>TTS</span></div>
          <div class="lat"><b>{latency.e2e_ms||0}</b><span>E2E</span></div>
        </div>
        {#if audioError}<div style="margin-top:8px; padding:7px; background:#1a0f14; border:1px solid #3a1a1a; border-radius:8px; font-size:11px; color:#ffb3b3">⚠️ {audioError}</div>{/if}
        <div class="kpi" style="margin-top:10px">
          <div>{sttPartial || '—'}</div>
          <div>{llmStreaming ? llmStreaming.slice(0,32)+'…' : '—'}</div>
        </div>
        <p class="subtle" style="margin-top:10px">Svelte 5 · AudioWorklet · Binary WS · barge-in · pre-roll 0.6s</p>
      </div>

      <div class="card">
        <h3>Stack</h3>
        <div style="font-size:12px; line-height:1.6; opacity:0.85">
          <div><b>STT</b> X-ASR 160ms-int8 (16k, zh+en) · <b>LLM</b> Qwen3.5-0.8B Q8_0 · <b>Embed</b> Granite-97M Q8 (CUDA, 384d) · <b>TTS</b> PrimeTTS v2 16k</div>
          <div class="subtle">Agent <b>Qwen-Agent</b> (3 tools) · SearXNG self-host + wttr.in + Bing · Tools: web_search · get_weather · get_current_datetime · honest search</div>
        </div>
      </div>
    </div>

    <div class="stack">
      <div class="card" style="display:flex; flex-direction:column; min-height:500px">
        <h3>Conversation <span style="margin-left:auto; font-size:11px; opacity:0.5; text-transform:none; letter-spacing:0">{chatHistory.length} msgs</span> <button class="ghost" style="padding:4px 8px; font-size:11px; margin-left:8px" onclick={clearChat}>Clear</button></h3>
        <div class="chat" id="chat">
          {#each chatHistory as m}
            {#if m.role==='assistant'}
              <div class="bubble {m.role} {m.streaming ? 'streaming' : ''} markdown">{@html renderMarkdown(m.text)}{#if m.streaming}<span style="opacity:0.6"> ▌</span>{/if}</div>
            {:else}
              <div class="bubble {m.role} {m.streaming ? 'streaming' : ''}">{m.text}{#if m.streaming}<span style="opacity:0.6"> ▌</span>{/if}</div>
            {/if}
          {:else}
            <div class="bubble system">No messages — hit <b>Listen</b> or type below.</div>
          {/each}
        </div>
        {#if toolStatus}<div style="margin-top:8px; padding:8px 10px; background:#0e1a14; border:1px solid #1e3326; border-radius:8px; font-size:12px; color:#8fd9b0; display:flex; gap:8px; align-items:center"><span style="animation:pulse 1.2s infinite">●</span> {toolStatus}</div>{/if}
        <div class="row">
          <input class="input" placeholder="Type a message…" bind:value={textInput} onkeydown={(e)=>{ if(e.key==='Enter' && textInput.trim()){ sendText(textInput.trim()); textInput=''; }}} />
          <button class="primary" onclick={()=>{ if(textInput.trim()){ sendText(textInput.trim()); textInput=''; }}}>Send</button>
        </div>
        <div class="tool-bar">
          <button class="tool-chip" onclick={()=>testSearch('What is the weather in Paris today?')}>Paris</button>
          <button class="tool-chip" onclick={()=>testSearch('Search latest AI news')}>News</button>
          <button class="tool-chip" onclick={()=>testSearch('Who is the president of France?')}>Who is…</button>
          <button class="tool-chip" onclick={()=>testSearch('今天是星期幾？')}>今天</button>
        </div>
        {#if lastSearchResults.length}
          <div style="margin-top:10px; padding:8px; background:rgba(0,0,0,0.2); border-radius:8px; max-height:140px; overflow:auto; border:1px solid #1e1e28">
            {#each lastSearchResults.slice(0,2) as r}
              <div style="font-size:11px; margin-bottom:6px; padding:6px; background:rgba(255,255,255,0.04); border-radius:6px">
                <a href={r.url} target="_blank" style="color:#8ea6ff; font-weight:600; text-decoration:none">{toTraditional(r.title)}</a><br/>
                <span style="opacity:0.6">{toTraditional(r.content.slice(0,120))}…</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>
  </div>

  <div style="text-align:center; font-size:11px; opacity:0.4; margin-top:16px">
    Silero VAD · X-ASR-int8 · Qwen3.5-0.8B Q8_0 · Granite-97M Q8 · PrimeTTS v2 · Qwen-Agent · <a href="https://github.com/vieenrose/voice-chat" style="color:inherit">GitHub</a>
  </div>
</div>
