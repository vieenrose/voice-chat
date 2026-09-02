/**
 * OpenAI Realtime protocol client for the speech-to-speech server.
 *
 * Speaks the subset the server actually implements (see its Realtime Engine
 * README): session.update, input_audio_buffer.append, conversation.item.create,
 * response.create and response.cancel outbound; speech start/stop, streaming
 * transcription, audio deltas and response lifecycle inbound.
 *
 * Callbacks rather than an event emitter, because there is exactly one consumer.
 */

// Where the server is, from the page's own origin.
//
//   http  -> the pipeline directly on :8765 (dev, or same machine)
//   https -> :8443, where a TLS terminator fronts it. A browser blocks ws:// from
//            an https page as mixed content, and the pipeline speaks plain ws, so
//            something has to terminate TLS. Here that is `tailscale serve --https
//            8443 http://127.0.0.1:8765`, which keeps the endpoint tailnet-only
//            while the page itself may be published more widely.
//
// Both are only defaults; the endpoint is editable in the UI.
const RT_TLS_PORT = 8443
const RT_PLAIN_PORT = 8765
const defaultUrl = () =>
  location.protocol === 'https:'
    ? `wss://${location.hostname}:${RT_TLS_PORT}/v1/realtime`
    : `ws://${location.hostname}:${RT_PLAIN_PORT}/v1/realtime`

export class RealtimeClient {
  constructor(handlers = {}) {
    this.h = handlers
    this.ws = null
    this.ready = false
  }

  connect(url = defaultUrl(), instructions = '') {
    return new Promise((resolve, reject) => {
      let ws
      try {
        ws = new WebSocket(url)
      } catch (e) {
        reject(e)
        return
      }
      this.ws = ws
      ws.binaryType = 'arraybuffer'

      ws.onopen = () => {
        this.ready = true
        this.send({
          type: 'session.update',
          session: {
            type: 'realtime',
            instructions,
            audio: {
              // No input format. AudioPCM.rate is Literal[24000] in the schema, so
              // declaring the mic's real 16k is rejected -- and a rejected field
              // fails the WHOLE session.update with "Unknown or invalid event".
              // Omitted means the server's own 16k pipeline rate, which is what we send.
              input: { turn_detection: { type: 'server_vad', interrupt_response: true } },
              // 24k is the rate Qwen3-TTS natively produces; without this the
              // server downsamples to 16k on the way out.
              output: { format: { type: 'audio/pcm', rate: 24000 } },
            },
          },
        })
        resolve()
      }
      ws.onerror = () => {
        if (!this.ready) reject(new Error(`cannot reach ${url}`))
        this.h.onError?.('WebSocket error')
      }
      ws.onclose = () => {
        this.ready = false
        this.h.onClose?.()
      }
      ws.onmessage = (ev) => {
        let d
        try {
          d = JSON.parse(ev.data)
        } catch {
          return
        }
        this._dispatch(d)
      }
    })
  }

  _dispatch(d) {
    const t = d.type
    switch (t) {
      case 'session.created':
      case 'session.updated':
        break

      case 'input_audio_buffer.speech_started':
        this.h.onSpeechStart?.()
        break
      case 'input_audio_buffer.speech_stopped':
        this.h.onSpeechStop?.()
        break

      case 'conversation.item.input_audio_transcription.delta':
        // NB: this delta is CUMULATIVE -- each one carries the whole transcript so
        // far ("欢迎" then "欢迎大家来" then ...), not the increment. Appending
        // them would produce "欢迎欢迎大家来欢迎大家来体验...".
        this.h.onUserPartial?.(d.delta || '')
        break
      case 'conversation.item.input_audio_transcription.completed':
        this.h.onUserFinal?.(d.transcript || '')
        break

      case 'response.created':
        this.h.onResponseStart?.()
        break
      case 'response.output_audio_transcript.delta':
        this.h.onAssistantText?.(d.delta || '')
        break
      case 'response.output_audio_transcript.done':
        // With an audio response the server sends the text on .done, once per
        // spoken chunk -- so these accumulate into the reply, they do not replace it.
        this.h.onAssistantText?.(d.transcript || '')
        break
      case 'response.output_text.delta':
        this.h.onAssistantText?.(d.delta || '')
        break
      case 'response.output_audio.delta':
        if (d.delta) this.h.onAudio?.(d.delta)
        break

      case 'response.done': {
        const r = d.response || {}
        const reason = (r.status_details || {}).reason || ''
        this.h.onResponseDone?.(r.status || 'completed', reason)
        break
      }
      case 'error':
        this.h.onError?.((d.error || {}).message || 'unknown error')
        break
      default:
        this.h.onOther?.(t, d)
    }
  }

  send(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj))
  }

  /** base64 PCM16 @ the server's 16k pipeline rate. */
  appendAudio(b64) {
    this.send({ type: 'input_audio_buffer.append', audio: b64 })
  }

  sendText(text) {
    this.send({
      type: 'conversation.item.create',
      item: { type: 'message', role: 'user', content: [{ type: 'input_text', text }] },
    })
    this.send({ type: 'response.create' })
  }

  cancel() {
    this.send({ type: 'response.cancel' })
  }

  close() {
    try {
      this.ws?.close()
    } catch {
      /* already gone */
    }
    this.ws = null
    this.ready = false
  }
}
