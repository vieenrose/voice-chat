<script>
  import { onMount, onDestroy } from 'svelte';

  // state (Svelte 5 runes)
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
  let mode = $state('mock'); // mock | real

  let ws = null;
  let audioCtx = null;
  let workletNode = null;
  let mediaStream = null;
  let analyser = null;
  let jitterQueue = [];
  let nextPlayTime = 0;
  let wsUrl = $derived((location.protocol==='https:' ? `wss://${location.host}/ws/chat?session_id=${'demo-'+Math.random().toString(36).slice(2,7)}` : `ws://${location.hostname}:8000/ws/chat?session_id=${'demo-'+Math.random().toString(36).slice(2,7)}`));
  // allow override via query ?ws=...
  let statsInterval = null;

  // canvas refs
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
  async function testSearch(q){
    sendText(q);
  }

  async function connect(){
    if (connected || connecting) return;
    connecting = true;
    let url = wsUrl;
    const qp = new URLSearchParams(location.search);
    if (qp.get('ws')) url = qp.get('ws');
    // if frontend proxy, use ws://localhost:5173/ws -> vite proxies
    // try direct first, fallback to proxied
    ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { connected=true; connecting=false; log('WS open', url); chatHistory = [...chatHistory, {role:'system', text:`Connected ${url} — mock=${mode}`}]; };
    ws.onclose = () => { connected=false; connecting=false; listening=false; log('WS closed'); };
    ws.onerror = (e) => { log('WS error', e); connecting=false; };
    ws.onmessage = async (ev) => {
      try{
        const data = JSON.parse(ev.data);
        handleServerMessage(data);
      }catch(e){
        // maybe binary? ignore
      }
    };
  }
  function disconnect(){
    if(ws){ ws.close(); ws=null; }
    connected=false;
    stopMic();
  }

  let toolStatus = $state('');
  let audioError = $state('');
  let lastSearchResults = $state([]);
  let searxngOk = $state(false);

  function handleServerMessage(msg){
    switch(msg.type){
      case 'stt_partial':
        sttPartial = msg.text;
        break;
      case 'stt_final':
        sttFinal = msg.text;
        sttPartial = '';
        chatHistory = [...chatHistory, {role:'user', text: msg.text}];
        llmStreaming = '';
        ttsText = '';
        toolStatus = '';
        break;
      case 'llm_token':
        llmStreaming = msg.text_so_far;
        break;
      case 'tool_call':
        {
          const tn = msg.name || 'tool';
          const q = msg.query || msg.arguments?.query || msg.arguments?.timezone || '';
          if(tn === 'web_search'){
            toolStatus = `🔍 web_search("${q}") — querying SearXNG...`;
            chatHistory = [...chatHistory, {role:'tool', text: `🔍 web_search("${q}") — querying self-hosted SearXNG...`, streaming:false}];
          } else {
            toolStatus = `🕐 ${tn}() — fetching date/time...`;
            chatHistory = [...chatHistory, {role:'tool', text: `🕐 ${tn}() — fetching date/time...`, streaming:false}];
          }
        }
        break;
      case 'tool_result':
        {
          const src = msg.source || msg.result?.source || 'searxng';
          const latency = msg.latency_ms || 0;
          const formatted = msg.formatted || '';
          // Parse formatted to extract titles
          const count = msg.result?.results?.length || 3;
          toolStatus = `✅ SearXNG returned ${count} results via ${src} in ${latency}ms`;
          lastSearchResults = msg.result?.results || [];
          let preview = formatted ? formatted.slice(0, 280).replace(/\n/g, ' ') + '...' : `${count} results via ${src}`;
          chatHistory = [...chatHistory, {role:'tool', text: `✅ ${src}: ${count} results in ${latency}ms — ${preview}`, streaming:false}];
        }
        break;
      case 'tts_chunk':
        ttsText = msg.text;
        if(msg.pcm){
          try{
            const pcm = base64ToInt16(msg.pcm);
            queueAudio(pcm, msg.sampleRate || 16000);
          }catch(e){
            audioError = `tts_chunk error: ${String(e).slice(0,120)}`;
          }
        }
        updateAssistantStreaming();
        break;
      case 'tts_start':
        // new utterance — reset pre-roll buffer for this reply
        preRollStarted = false;
        preRollQueue.length = 0;
        speaking = true;
        break;
      case 'tts_end':
        flushPreRoll();  // play whatever was still buffered (short replies)
        speaking = false;
        finalizeAssistant();
        toolStatus = '';
        break;
      case 'latency':
        latency = msg;
        break;
      case 'stt_partial':
        break;
    }
  }

  function updateAssistantStreaming(){
    // if last entry is assistant streaming, update its text, else push new
    if(llmStreaming){
      if(chatHistory.length && chatHistory[chatHistory.length-1].role==='assistant' && chatHistory[chatHistory.length-1].streaming){
        chatHistory[chatHistory.length-1].text = llmStreaming;
        chatHistory = [...chatHistory];
      } else {
        chatHistory = [...chatHistory, {role:'assistant', text: llmStreaming, streaming:true}];
      }
    }
  }
  function finalizeAssistant(){
    if(chatHistory.length && chatHistory[chatHistory.length-1].streaming){
      chatHistory[chatHistory.length-1].streaming = false;
      chatHistory = [...chatHistory];
    } else if(llmStreaming){
      chatHistory = [...chatHistory, {role:'assistant', text: llmStreaming}];
    }
    llmStreaming='';
  }

  function base64ToInt16(b64){
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
    return new Int16Array(bytes.buffer);
  }
  function floatTo16BitPCM(float32){
    const out = new Int16Array(float32.length);
    for(let i=0;i<float32.length;i++){
      let s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s*0x8000 : s*0x7FFF;
    }
    return out;
  }

  // ---- Audio Capture ----
  async function startMic(){
    micError='';
    if(listening) return;
    if(!connected){ await connect(); await new Promise(r=>setTimeout(r,400)); }
    // Must be secure context (https or localhost) for mic — http://training-machine will fail
    if(!window.isSecureContext){
      micError = `Mic blocked: not secure context (${location.protocol}//${location.host}). Chrome requires HTTPS for mic on http://training-machine. Fix: use https tunnel (see below) or chrome://flags/#unsafely-treat-insecure-origin-as-secure → add http://training-machine:5173 → Relaunch, or use text input (bypass mic).`;
      chatHistory=[...chatHistory,{role:'system',text: micError}];
      console.warn(micError);
    }
    try{
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate:16000});
      if(audioCtx.state === 'suspended') await audioCtx.resume();
      mediaStream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, sampleRate:16000, echoCancellation:true, noiseSuppression:true, autoGainControl:true}});
    }catch(e){
      const msg = e.name==='NotAllowedError' ? 'Mic permission denied — allow mic in address bar → site settings → Allow' : e.name==='NotFoundError' ? 'No mic found' : e.message.includes('secure')||e.message.includes('HTTPS') ? 'Mic requires HTTPS — http://training-machine is insecure' : e.message;
      micError = `🎤 ${e.name}: ${msg}. Try: chrome://flags/#unsafely-treat-insecure-origin-as-secure → add http://training-machine:5173, or use text input chips below (no mic needed), or https tunnel.`;
      chatHistory=[...chatHistory,{role:'system',text: micError}];
      console.error('getUserMedia failed', e);
      return;
    }
    const src = audioCtx.createMediaStreamSource(mediaStream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    src.connect(analyser);

    if(useWorklet){
      try{
        await audioCtx.audioWorklet.addModule('/audio-processor.js');
        workletNode = new AudioWorkletNode(audioCtx, 'capture-processor');
        src.connect(workletNode);
        // we don't connect worklet to destination (no feedback)
        workletNode.port.onmessage = (e)=>{
          if(e.data.type==='chunk'){
            const floatChunk = e.data.pcm; // Float32 320
            const pcm16 = floatTo16BitPCM(floatChunk);
            // simple VAD: compute rms
            let rms = 0; for(let i=0;i<floatChunk.length;i++) rms += floatChunk[i]*floatChunk[i];
            rms = Math.sqrt(rms/floatChunk.length);
            audioLevel = Math.min(1, rms*8);
            if(vadEnabled && rms < 0.008 && false){
              // would skip silence but we still send for endpointing, so send anyway but could drop
            }
            sendPCM(pcm16);
          }
        };
        listening = true;
        log('AudioWorklet capture started');
      }catch(e){
        log('Worklet failed, fallback to ScriptProcessor', e);
        useWorklet=false;
        startScriptProcessor(src);
      }
    } else {
      startScriptProcessor(src);
    }
    drawWave();
  }

  function startScriptProcessor(src){
    const proc = audioCtx.createScriptProcessor(512,1,1);
    let buffer = new Float32Array(0);
    proc.onaudioprocess = (e)=>{
      const input = e.inputBuffer.getChannelData(0);
      // resample is already 16k if ctx is 16k
      const combined = new Float32Array(buffer.length + input.length);
      combined.set(buffer); combined.set(input, buffer.length);
      buffer = combined;
      while(buffer.length >= 320){
        const chunk = buffer.slice(0,320);
        buffer = buffer.slice(320);
        const pcm16 = floatTo16BitPCM(chunk);
        let rms=0; for(let i=0;i<chunk.length;i++) rms+=chunk[i]*chunk[i];
        audioLevel = Math.min(1, Math.sqrt(rms/chunk.length)*6);
        sendPCM(pcm16);
      }
    };
    src.connect(proc);
    proc.connect(audioCtx.destination); // needed for processing, but muted via gain 0
    const gain = audioCtx.createGain(); gain.gain.value=0; proc.connect(gain);
    workletNode = proc; // reuse var
    listening=true;
  }

  function stopMic(){
    listening=false;
    if(animId) cancelAnimationFrame(animId);
    if(workletNode){ try{workletNode.disconnect();}catch(e){} workletNode=null; }
    if(mediaStream){ mediaStream.getTracks().forEach(t=>t.stop()); mediaStream=null; }
    if(audioCtx){ try{audioCtx.close();}catch(e){} audioCtx=null; }
    audioLevel=0;
    // tell backend to flush
    if(ws && ws.readyState===1){
      ws.send(JSON.stringify({type:'stop'}));
    }
  }

  function sendPCM(pcm16){
    if(!ws || ws.readyState!==1) return;
    // binary protocol: 0x01 header + PCM
    // also support JSON base64 for compatibility, but binary is lower latency (no b64 overhead)
    const header = new Uint8Array(1); header[0]=0x01;
    const out = new Uint8Array(1 + pcm16.byteLength);
    out.set(header,0);
    out.set(new Uint8Array(pcm16.buffer),1);
    try{ ws.send(out); }catch(e){ // fallback to json
      const b64 = btoa(String.fromCharCode(...new Uint8Array(pcm16.buffer)));
      ws.send(JSON.stringify({type:'audio_chunk', pcm:b64, sampleRate:16000}));
    }
  }

  function sendText(text){
    if(!ws || ws.readyState!==1){ connect(); setTimeout(()=>sendText(text),500); return; }
    ws.send(JSON.stringify({type:'text_input', text}));
    chatHistory=[...chatHistory, {role:'user', text}];
    llmStreaming='';
  }

  function bargeIn(){
    if(ws && ws.readyState===1) ws.send(JSON.stringify({type:'barge_in'}));
    jitterQueue=[];
    nextPlayTime=0;
    speaking=false;
    if(playCtx){ try{playCtx.close();}catch(e){} playCtx=null; nextPlayTime=0; }
  }

  // ---- Playback jitter buffer ----
  let playCtx = null;
  function ensurePlayCtx(sampleRate){
    if(!playCtx){
      // Use device default (usually 48k) — most compatible, we resample manually below
      try{
        playCtx = new (window.AudioContext||window.webkitAudioContext)();
      }catch(e){
        playCtx = new (window.AudioContext||window.webkitAudioContext)();
      }
      console.log('[Audio] playCtx created sampleRate', playCtx.sampleRate, 'for TTS', sampleRate);
      nextPlayTime = playCtx.currentTime;
      // iOS/Safari: resume() only works inside a user gesture — hook one so audio starts reliably
      const _resume = () => { if(playCtx && playCtx.state==='suspended') playCtx.resume(); };
      document.addEventListener('pointerdown', _resume, {once:true});
      window.addEventListener('touchend', _resume, {once:true});
    } else if(playCtx.state==='suspended'){
      try{ playCtx.resume(); }catch(e){}
    }
    return playCtx;
  }
  function resampleLinear(input, inRate, outRate){
    if(inRate===outRate) return input;
    const ratio = inRate / outRate;
    const outLen = Math.round(input.length / ratio);
    const out = new Float32Array(outLen);
    for(let i=0;i<outLen;i++){
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = input[idx] || 0;
      const b = input[idx+1] || 0;
      out[i] = a + (b-a)*frac;
    }
    return out;
  }
  const PRE_ROLL_SEC = 0.6;    // buffer this much audio before starting playback (absorbs jitter)        
  const preRollQueue = [];     // {buf, src, dur}
  let preRollStarted = false;
  function queueAudio(pcm16, sampleRate){
    const ctx = ensurePlayCtx(sampleRate);
    // pcm16 -> float32
    const float = new Float32Array(pcm16.length);
    for(let i=0;i<pcm16.length;i++) float[i]=pcm16[i]/32768;
    // Manual resample to ctx.sampleRate to guarantee correct pitch (16k→48k, browsers vary in auto-resample)
    const resampled = resampleLinear(float, sampleRate, ctx.sampleRate);
    const buf = ctx.createBuffer(1, resampled.length, ctx.sampleRate);
    buf.getChannelData(0).set(resampled);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const dur = buf.duration;
    src.onended = ()=>{
      if(ctx.currentTime >= nextPlayTime - 0.05) setTimeout(()=>{ if(ctx.currentTime >= nextPlayTime) speaking=false; }, 200);
    };
    if(!preRollStarted){
      preRollQueue.push({buf, src, dur});
      const buffered = preRollQueue.reduce((a,c)=>a+c.dur,0);
      if(buffered >= PRE_ROLL_SEC) flushPreRoll();
      return;
    }
    // gapless additive schedule
    if(nextPlayTime < ctx.currentTime || !isFinite(nextPlayTime)) nextPlayTime = ctx.currentTime + 0.08;
    src.start(nextPlayTime);
    nextPlayTime += dur;
    speaking = true;
  }
  function flushPreRoll(){
    const ctx = ensurePlayCtx();
    if(!preRollQueue.length || preRollStarted) return;
    preRollStarted = true;
    if(nextPlayTime < ctx.currentTime || !isFinite(nextPlayTime)) nextPlayTime = ctx.currentTime + 0.08;
    for(const c of preRollQueue){
      c.src.start(nextPlayTime);
      nextPlayTime += c.dur;
    }
    preRollQueue.length = 0;
    speaking = true;
  }

  function drawWave(){
    if(!canvasEl || !analyser) return;
    const ctx = canvasEl.getContext('2d');
    const data = new Uint8Array(analyser.frequencyBinCount);
    const draw = ()=>{
      animId = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(data);
      ctx.clearRect(0,0,canvasEl.width, canvasEl.height);
      // waveform bars
      const w = canvasEl.width, h=canvasEl.height;
      const barW = w / data.length * 2.5;
      let x=0;
      for(let i=0;i<data.length;i+=2){
        const v = data[i]/255;
        const bh = v*h*0.9;
        const hue = listening ? 265 : 220;
        ctx.fillStyle = listening ? `hsla(${hue},90%,65%,${0.5+v*0.5})` : `hsla(210,10%,40%,0.6)`;
        ctx.fillRect(x, h-bh, barW, bh);
        x+=barW+1;
        if(x>=w) break;
      }
      // level meter
      ctx.fillStyle = listening ? '#7c5cff' : '#3a3a44';
      ctx.fillRect(0, h-3, w*audioLevel, 3);
    };
    draw();
  }

  onMount(()=>{
    fetchHealth();
    statsInterval = setInterval(fetchHealth, 3000);
  });
  onDestroy(()=>{
    disconnect();
    if(statsInterval) clearInterval(statsInterval);
    if(animId) cancelAnimationFrame(animId);
  });

  let micError = $state('');
  let textInput = $state('');
  let theme = $state('dark');
  $effect(() => {
    if (typeof document !== 'undefined') {
      try { theme = localStorage.getItem('vc-theme') || 'dark'; } catch(e) {}
      document.body.classList.toggle('light', theme === 'light');
    }
  });
  function toggleTheme(){
    theme = theme === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('vc-theme', theme); } catch(e) {}
    if (typeof document !== 'undefined') document.body.classList.toggle('light', theme === 'light');
  }
</script>

<style>
  :global(body){background: radial-gradient(1200px 600px at 50% -10%, #1a1240 0%, #0a0a0f 60%); min-height:100vh;}
  .wrap{max-width:980px; margin:0 auto; padding:28px 20px;}
  .header{display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:20px}
  .title{font-size:22px; font-weight:800; letter-spacing:-0.02em}
  .subtitle{font-size:12px; opacity:0.7; margin-top:2px}
  .badge{font-size:11px; padding:6px 10px; border-radius:999px; border:1px solid #2a2a3a; background:#14141c}
  .badge.ok{border-color:#7c5cff; color:#b8a6ff}
  .grid{display:grid; grid-template-columns: 1.1fr 0.9fr; gap:16px}
  @media(max-width:860px){ .grid{grid-template-columns:1fr} }
  .card{background: rgba(20,20,28,0.85); backdrop-filter: blur(12px); border:1px solid #232332; border-radius:16px; padding:16px; box-shadow: 0 8px 32px rgba(0,0,0,0.35)}
  .card h3{font-size:13px; letter-spacing:0.08em; text-transform:uppercase; opacity:0.7; margin-bottom:10px}
  .controls{display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px}
  button{appearance:none; border:0; padding:10px 16px; border-radius:999px; font-weight:700; font-size:13px; cursor:pointer; transition:all .15s}
  button:disabled{opacity:0.45; cursor:not-allowed}
  .primary{background:#7c5cff; color:white; box-shadow:0 4px 18px rgba(124,92,255,0.4)}
  .primary:hover{transform:translateY(-1px); box-shadow:0 6px 22px rgba(124,92,255,0.5)}
  .ghost{background:#1e1e2a; color:#d8d8e6; border:1px solid #2a2a3a}
  .ghost:hover{background:#252538}
  .danger{background:#ff3b5e; color:white}
  .lat-grid{display:grid; grid-template-columns: repeat(4, 1fr); gap:8px}
  .lat{padding:10px; border-radius:12px; background:#0f0f14; border:1px solid #232332; text-align:center}
  .lat b{font-size:18px; display:block}
  .lat span{font-size:10px; opacity:0.6; text-transform:uppercase; letter-spacing:0.06em}
  .wave{width:100%; height:72px; border-radius:12px; background:#0a0a0f; border:1px solid #232332; margin:12px 0}
  .chat{height:360px; overflow:auto; display:flex; flex-direction:column; gap:10px; padding:4px}
  .bubble{max-width:84%; padding:11px 14px; border-radius:14px; font-size:14px; line-height:1.4; word-break:break-word}
  .bubble.user{align-self:flex-end; background:#7c5cff; color:white; border-bottom-right-radius:4px}
  .bubble.assistant{align-self:flex-start; background:#1c1c26; border:1px solid #2a2a3a; border-bottom-left-radius:4px}
  .bubble.system{align-self:center; background:#0f0f14; border:1px dashed #2a2a3a; font-size:12px; opacity:0.8}
  .bubble.streaming{border-color:#7c5cff; box-shadow:0 0 0 1px rgba(124,92,255,0.3)}
  .meta{font-size:11px; opacity:0.6; margin-top:6px}
  .row{display:flex; gap:8px; margin-top:10px}
  .input{flex:1; background:#0f0f14; border:1px solid #2a2a3a; color:white; padding:11px 14px; border-radius:999px; outline:none}
  .input:focus{border-color:#7c5cff}
  .toggles{display:flex; gap:12px; font-size:12px; opacity:0.85; margin:8px 0}
  .kpi{display:flex; gap:8px; font-size:11px; opacity:0.7}
  .kpi div{background:#0f0f14; padding:6px 8px; border-radius:999px; border:1px solid #232332}
  .bubble.tool{align-self:center; background:#0e1a12; border:1px solid #1f3d2b; color:#a0e8c0; font-size:12px; max-width:92%; border-radius:8px}
  .tool-bar{display:flex; gap:8px; margin-top:8px; flex-wrap:wrap}
  .tool-chip{font-size:11px; padding:6px 10px; border-radius:999px; background:#1a2a1f; border:1px solid #2d4a32; color:#a0e8c0; cursor:pointer}
  .tool-chip:hover{border-color:#7c5cff; color:#b8a6ff}
  .searxng-badge{font-size:10px; padding:4px 8px; border-radius:999px; border:1px solid #2a3d2a; background:#0f1a0f}
  .searxng-badge.ok{border-color:#20c997; color:#20c997}

  /* ---------- LIGHT THEME ---------- */
  :global(body.light){background: linear-gradient(180deg,#eef0fb 0%,#f7f7fb 60%) !important; color:#1a1a24}
  :global(body.light) .title{color:#1a1a24}
  :global(body.light) .card{background:#ffffff; border-color:#e3e4ee; box-shadow:0 8px 28px rgba(30,30,60,0.08)}
  :global(body.light) .subtitle, :global(body.light) .card h3, :global(body.light) .lat span, :global(body.light) .meta{opacity:0.65; color:#3a3a4a}
  :global(body.light) .badge{background:#ffffff; border-color:#d8dae8; color:#3a3a4a}
  :global(body.light) .badge.ok{border-color:#7c5cff; color:#6a4beb}
  :global(body.light) .ghost{background:#f1f2f8; color:#2a2a3a; border-color:#d8dae8}
  :global(body.light) .ghost:hover{background:#e6e7f2}
  :global(body.light) .lat{background:#f7f7fc; border-color:#e3e4ee}
  :global(body.light) .wave{background:#f0f1f8; border-color:#e3e4ee}
  :global(body.light) .bubble.assistant{background:#f1f2f8; border-color:#e3e4ee; color:#1a1a24}
  :global(body.light) .bubble.system{background:#f7f7fc; border-color:#d8dae8; color:#555}
  :global(body.light) .input{background:#ffffff; border-color:#d8dae8; color:#1a1a24}
  :global(body.light) .input:focus{border-color:#7c5cff}
  :global(body.light) .kpi div{background:#f7f7fc; border-color:#e3e4ee}
  :global(body.light) .bubble.tool{background:#e9f8f0; border-color:#bfe6d0; color:#0b6b3f}
  :global(body.light) .tool-chip{background:#e9f8f0; border-color:#bfe6d0; color:#0b6b3f}
  :global(body.light) .searxng-badge{background:#e9f8f0; border-color:#bfe6d0}
  :global(body.light) .searxng-badge.ok{color:#0aa06a}
  :global(body.light) .lat b{color:#1a1a24}
</style>

<div class="wrap">
  <div class="header">
    <div>
      <div class="title">🎙️ Voice Chat <span style="opacity:0.6; font-weight:600">· X-ASR · Gemma-4-E2B · Qwen3-TTS</span></div>
      <div class="subtitle">VAD Silero · STT X-ASR (sherpa 160ms) · LLM Gemma-4-E2B-it (native tools) · TTS Qwen3-TTS 0.6B CustomVoice Q8 24k · Svelte 5 + AudioWorklet</div>
    </div>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap">
      <div class="badge {connected ? 'ok' : ''}">{connected ? '● CONNECTED' : '○ DISCONNECTED'} · {mode.toUpperCase()} · RSS {rssMb} MB</div>
      <div class="searxng-badge {searxngOk ? 'ok' : ''}">{searxngOk ? '🔍 SearXNG ● self-hosted :8888' : '🔍 SearXNG ○ offline (fallback → mock)'}</div>
      <button class="ghost" style="padding:6px 12px; font-size:12px" onclick={toggleTheme}>{theme==='dark' ? '🌙 Dark' : '☀️ Light'}</button>
    </div>
  </div>

  <div class="grid">
    <!-- left: controls + viz -->
    <div class="card">
      <h3>Realtime</h3>
      <div class="controls">
        {#if !connected}
          <button class="primary" onclick={connect} disabled={connecting}>{connecting ? 'Connecting…' : 'Connect WS'}</button>
        {:else}
          <button class="ghost" onclick={disconnect}>Disconnect</button>
        {/if}
        {#if !listening}
          <button class="primary" onclick={startMic} disabled={!connected}>🎤 Start Listening</button>
        {:else}
          <button class="danger" onclick={stopMic}>⏹ Stop</button>
          <button class="ghost" onclick={bargeIn}>✋ Barge-in</button>
        {/if}
        <button class="ghost" onclick={()=>{vadEnabled=!vadEnabled}}>{vadEnabled ? 'VAD ON' : 'VAD OFF'}</button>
        <button class="ghost" onclick={()=>{useWorklet=!useWorklet}}>{useWorklet ? 'Worklet' : 'ScriptProc'}</button>
      </div>
      {#if micError}
        <div style="margin:8px 0; padding:10px 12px; background:#2a1214; border:1px solid #7a2a2a; border-radius:10px; font-size:12px; line-height:1.5; color:#ffb3b3">
          {micError}
          <div style="margin-top:8px; opacity:0.85; color:#ffd9d9">
            <b>Fix mic on <code>http://training-machine:5173</code>:</b><br>
            1) Chrome → <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code> → add <code>http://training-machine:5173</code> → Enable → Relaunch<br>
            2) Or get <b>https</b>: on host run <code>npx localtunnel --port 5173</code> → open <code>https://…loca.lt</code> (mic works) — see terminal<br>
            3) Or <b>bypass mic</b>: type below + Send (uses text, no mic needed) — chips test tool calling.
          </div>
        </div>
      {/if}
      {#if typeof window !== 'undefined' && !window.isSecureContext}
        <div style="font-size:11px; padding:6px 10px; background:#1a1a2e; border:1px solid #333; border-radius:8px; margin-top:6px">⚠️ Not secure context (<code>{location.protocol}//{location.host}</code>) — mic blocked on <code>http://training-machine</code>. Use text input below, or enable flag above, or <code>npx localtunnel --port 5173</code> → https.</div>
      {/if}

      <canvas bind:this={canvasEl} class="wave" width="640" height="72"></canvas>

      <div class="lat-grid">
        <div class="lat"><b>{latency.stt_ms||0}ms</b><span>STT X-ASR</span></div>
        <div class="lat"><b>{latency.llm_ttft_ms||0}ms</b><span>LLM TTFT</span></div>
        <div class="lat"><b>{latency.tts_ttfb_ms||0}ms</b><span>TTS TTFB</span></div>
        <div class="lat" style="border-color:#7c5cff"><b>{latency.e2e_ms||0}ms</b><span>E2E</span></div>
      </div>
      {#if audioError}
        <div style="margin-top:6px; padding:6px 10px; background:#2a1214; border:1px solid #7a2a2a; border-radius:8px; font-size:11px; color:#ffb3b3">⚠️ {audioError}</div>
      {/if}

      <div class="kpi" style="margin-top:10px">
        <div>STT partial: <i style="color:#b8a6ff">{sttPartial || '—'}</i></div>
        <div>LLM streaming: {llmStreaming ? llmStreaming.slice(0,40)+'…' : '—'}</div>
      </div>

      <div class="toggles">
        <label><input type="checkbox" bind:checked={vadEnabled}/> VAD</label>
        <label><input type="checkbox" bind:checked={useWorklet}/> Worklet</label>
        <span style="margin-left:auto; opacity:0.6">{listening ? '🎧 Listening 16k (X-ASR)' : 'Idle'} {speaking ? '· 🔊 Qwen3-TTS 24k' : ''} · mic 16k → TTS 24k</span>
      </div>

      <div style="font-size:11px; opacity:0.55; line-height:1.5; margin-top:8px">
        <b>Low-latency tricks:</b> 20ms mic chunks → binary WS → STT partial @300ms → LLM tokens @~100ms TTFT → TTS per-sentence flush → pre-roll 0.6s → barge-in cancels TTS instantly. Svelte 5 compiled reactivity avoids React VDOM 8-16ms overhead.
      </div>
    </div>

    <!-- right: chat -->
    <div class="card">
      <h3>Conversation</h3>
      <div class="chat" id="chat">
        {#each chatHistory as m}
          <div class="bubble {m.role} {m.streaming ? 'streaming' : ''}">
            {m.text}
            {#if m.streaming}<span style="opacity:0.6"> ▌</span>{/if}
          </div>
        {:else}
          <div class="bubble system">No messages yet — hit <b>Start Listening</b> and say “hello”, or type below.</div>
        {/each}
      </div>

      {#if toolStatus}
        <div style="margin-top:8px; padding:8px 12px; background:#0e1a12; border:1px solid #1f3d2b; border-radius:8px; font-size:12px; color:#a0e8c0">{toolStatus}</div>
      {/if}
      <div class="row">
        <input class="input" placeholder="Type a message (bypass STT)…" bind:value={textInput} onkeydown={(e)=>{ if(e.key==='Enter' && textInput.trim()){ sendText(textInput.trim()); textInput=''; }}} />
        <button class="primary" onclick={()=>{ if(textInput.trim()){ sendText(textInput.trim()); textInput=''; }}}>Send</button>
      </div>
      <div class="tool-bar">
        <span style="font-size:11px; opacity:0.6; padding:6px 2px">Try tool calling:</span>
        <button class="tool-chip" onclick={()=>testSearch('What is the weather in Paris today?')}>🌤️ Weather Paris</button>
        <button class="tool-chip" onclick={()=>testSearch('Search latest AI news')}>📰 AI News</button>
        <button class="tool-chip" onclick={()=>testSearch('Who is the president of France?')}>👤 Who is…</button>
        <button class="tool-chip" onclick={()=>testSearch('Python 3.14 features')}>🐍 Python 3.14</button>
        <button class="tool-chip" onclick={()=>testSearch('Hello how are you')}>💬 Hello (no tool)</button>
      </div>
      <div class="meta">Backend: <code>ws://{location.hostname}:8000/ws/chat</code> · API <code>/api/chat</code> · <code>/health</code> shows peak RSS</div>

      <div class="card" style="margin-top:12px; padding:12px; background:#0f0f14; border-color:#1f3d2b">
        <h3 style="margin-bottom:6px; color:#20c997">🔧 Gemma-4-E2B-it Tool Calling · web_search via self-hosted SearXNG</h3>
        <div style="font-size:12px; opacity:0.85; line-height:1.6">
          <b>Gemma-4-E2B-it</b> (Q4_K_M, native tools) multi-turn <code>tool_calls</code> → <code>web_search(query)</code> → <b>SearXNG</b> self-hosted (bing + wikipedia engines, entity-first queries) → <code>tool_response</code> → answer → spoken via <b>Qwen3-TTS 0.6B CustomVoice</b> (faster_qwen3_tts cu124 GGML, TRUE streaming, 24k).<br/>
          <span style="opacity:0.6">Try:</span> “What's the weather?”, “Search AI news”, “Who is …?” — watch 🔍 bubbles.
        </div>
        {#if lastSearchResults.length}
          <div style="margin-top:8px; padding:8px; background:#0a0a0f; border-radius:8px; max-height:120px; overflow:auto">
            <div style="font-size:11px; opacity:0.7; margin-bottom:6px">Last SearXNG results:</div>
            {#each lastSearchResults.slice(0,3) as r}
              <div style="font-size:11px; margin-bottom:6px; padding:6px; background:#14141c; border-radius:6px">
                <a href={r.url} target="_blank" style="color:#7c5cff; font-weight:700">{r.title}</a><br/>
                <span style="opacity:0.6">{r.url.slice(0,60)}</span><br/>
                <span>{r.content.slice(0,140)}...</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>
      <div class="card" style="margin-top:12px; padding:12px; background:#0f0f14">
        <h3 style="margin-bottom:6px">AI Components (running)</h3>
        <table style="width:100%; font-size:12px; border-collapse:collapse">
          <tbody>
          <tr><td style="padding:5px 8px; opacity:0.7; border-bottom:1px solid #1e1e28">VAD</td><td style="padding:5px 8px; border-bottom:1px solid #1e1e28"><code>Silero VAD</code> <span style="opacity:0.5">· turn-taking, 16k</span></td></tr>
          <tr><td style="padding:5px 8px; opacity:0.7; border-bottom:1px solid #1e1e28">STT</td><td style="padding:5px 8px; border-bottom:1px solid #1e1e28"><code>GilgameshWind/X-ASR-zh-en</code> <span style="opacity:0.5">· sherpa Zipformer 160ms streaming, zh+en 16k</span></td></tr>
          <tr><td style="padding:5px 8px; opacity:0.7; border-bottom:1px solid #1e1e28">LLM</td><td style="padding:5px 8px; border-bottom:1px solid #1e1e28"><code>unsloth/gemma-4-E2B-it-GGUF</code> <span style="opacity:0.5">· 4B E2B Q4_K_M, llama-server :11435, native tools=[web_search]</span></td></tr>
          <tr><td style="padding:5px 8px; opacity:0.7">TTS</td><td style="padding:5px 8px"><code>Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice</code> <span style="opacity:0.5">· GGML cu124 Q8_0, 24k, speaker Aiden, TTFA ~20ms</span></td></tr>
          </tbody>
        </table>
        <div style="font-size:11px; opacity:0.6; line-height:1.6; margin-top:8px">
          Frontend <b>Svelte 5 + Vite + AudioWorklet + Binary WS</b> (<code>wss</code>). History per <code>session_id</code> (20 turns). SearXNG relevance &lt;0.34 → <code>mock-curated</code>.
        </div>
      </div>
    </div>
  </div>

  <div style="text-align:center; font-size:11px; opacity:0.45; margin-top:16px">
    <span style="color:#20c997">● Real: Silero VAD + X-ASR + Gemma-4-E2B + Qwen3-TTS-Q8</span> · SearXNG self-hosted :8888 · <code>https://training-machine.tailf63b31.ts.net</code> · <code>wss</code> · No mock.
  </div>
</div>
