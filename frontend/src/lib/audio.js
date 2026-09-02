/**
 * Mic capture and speaker playback for the Realtime client.
 *
 * Capture runs an AudioContext pinned to 16 kHz -- the rate the server's VAD and
 * Paraformer expect -- so the browser does the resampling from whatever the
 * device gives us and there is no hand-rolled resampler to get wrong.
 *
 * Playback is a scheduled queue rather than one-shot plays: PCM chunks arrive
 * faster than real time, so each is started at the running play head to
 * concatenate seamlessly. The head is what makes barge-in audible -- dropping
 * queued audio is not enough, the already-scheduled sources have to be stopped.
 */

function floatToPCM16(f32) {
  const out = new Int16Array(f32.length)
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]))
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  return out
}

function bytesToBase64(bytes) {
  let bin = ''
  const CH = 0x8000 // chunked: String.fromCharCode blows the stack on big arrays
  for (let i = 0; i < bytes.length; i += CH) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH))
  }
  return btoa(bin)
}

function base64ToInt16(b64) {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1)
}

export class MicCapture {
  constructor({ sampleRate = 16000, onChunk, onLevel } = {}) {
    this.sampleRate = sampleRate
    this.onChunk = onChunk
    this.onLevel = onLevel
    this.ctx = null
    this.stream = null
    this.node = null
  }

  get active() {
    return !!this.node
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,   // without this the mic hears the reply and barges in on itself
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    this.ctx = new AudioContext({ sampleRate: this.sampleRate })
    if (this.ctx.state === 'suspended') await this.ctx.resume()
    await this.ctx.audioWorklet.addModule('/audio-processor.js')

    const src = this.ctx.createMediaStreamSource(this.stream)
    this.node = new AudioWorkletNode(this.ctx, 'capture-processor')
    this.node.port.onmessage = (e) => {
      if (e.data?.type !== 'chunk') return
      const f32 = e.data.pcm
      let peak = 0
      for (let i = 0; i < f32.length; i++) {
        const v = Math.abs(f32[i])
        if (v > peak) peak = v
      }
      this.onLevel?.(peak)
      this.onChunk?.(bytesToBase64(new Uint8Array(floatToPCM16(f32).buffer)))
    }
    src.connect(this.node)
    // Worklets are only pulled when connected to a destination. A zero-gain sink
    // keeps process() running without routing the mic to the speaker.
    const mute = this.ctx.createGain()
    mute.gain.value = 0
    this.node.connect(mute).connect(this.ctx.destination)
  }

  stop() {
    try {
      this.node?.disconnect()
    } catch { /* not connected */ }
    this.node = null
    this.stream?.getTracks().forEach((t) => t.stop())
    this.stream = null
    this.ctx?.close()
    this.ctx = null
  }
}

export class Playback {
  constructor({ sampleRate = 24000 } = {}) {
    this.sampleRate = sampleRate
    this.ctx = null
    this.head = 0
    this.sources = new Set()
    this.onEnded = null
  }

  _ensure() {
    if (!this.ctx) this.ctx = new AudioContext({ sampleRate: this.sampleRate })
    if (this.ctx.state === 'suspended') this.ctx.resume()
    return this.ctx
  }

  /** Queue one base64 PCM16 chunk. Returns its duration in seconds. */
  push(b64) {
    const ctx = this._ensure()
    const pcm = base64ToInt16(b64)
    if (!pcm.length) return 0
    const buf = ctx.createBuffer(1, pcm.length, this.sampleRate)
    const ch = buf.getChannelData(0)
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768

    const src = ctx.createBufferSource()
    src.buffer = buf
    src.connect(ctx.destination)
    // A small floor keeps the first chunk from being scheduled in the past on a
    // busy main thread, which silently drops it.
    const at = Math.max(ctx.currentTime + 0.02, this.head)
    src.start(at)
    this.head = at + buf.duration
    this.sources.add(src)
    src.onended = () => {
      this.sources.delete(src)
      if (!this.sources.size) this.onEnded?.()
    }
    return buf.duration
  }

  /** Barge-in: stop what is already scheduled, not merely what is not yet queued. */
  flush() {
    for (const s of this.sources) {
      try {
        s.stop()
      } catch { /* already ended */ }
    }
    this.sources.clear()
    this.head = this.ctx ? this.ctx.currentTime : 0
  }

  get playing() {
    return this.sources.size > 0
  }

  close() {
    this.flush()
    this.ctx?.close()
    this.ctx = null
  }
}
