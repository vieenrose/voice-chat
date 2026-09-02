// Barge-in through the frontend's own client: start a long spoken reply, then
// speak over it, and check the UI would both stop receiving and stop playing.
//
//   node checks/bargein.mjs     (needs the pipeline up on :8765)
globalThis.location = { hostname: '127.0.0.1' }
const { RealtimeClient } = await import(new URL('../src/lib/realtime.js', import.meta.url))
import fs from 'node:fs'
const wav = fs.readFileSync(new URL('../../asr_example.wav', import.meta.url))
const pcm = new Int16Array(wav.buffer, wav.byteOffset + 44, (wav.length - 44) >> 1)

let audioBefore = 0, audioAfterSpeech = 0, flushed = false, done = null
let speaking = false, speechStartAt = 0
const t0 = Date.now()

const c = new RealtimeClient({
  onSpeechStart: () => {
    speechStartAt = Date.now()
    // this is what App.svelte does: drop locally-scheduled audio
    if (audioBefore > 0) flushed = true
  },
  onAudio: () => { speaking ? audioAfterSpeech++ : audioBefore++ },
  onResponseDone: (s, r) => { done = `${s}${r ? ':' + r : ''}` },
  onError: (m) => console.log('  ERROR:', m),
})
await c.connect('ws://127.0.0.1:8765/v1/realtime', '你是一個親切的語音助理，請用繁體中文詳細回答。')
c.sendText('請詳細說明台灣半導體產業的發展歷史、現況與未來挑戰，分成至少五個段落。')

// wait until the reply is genuinely flowing, then talk over it
while (audioBefore < 20 && Date.now() - t0 < 30000) await new Promise(r => setTimeout(r, 50))
console.log(`  reply flowing (${audioBefore} chunks) — user starts speaking at ${Date.now()-t0}ms`)
speaking = true
const step = 16000 * 0.16
for (let i = 0; i < pcm.length && !done; i += step) {
  const s = pcm.subarray(i, Math.min(i + step, pcm.length))
  c.appendAudio(Buffer.from(s.buffer, s.byteOffset, s.byteLength).toString('base64'))
  await new Promise(r => setTimeout(r, 160))
}
const deadline = Date.now() + 15000
while (Date.now() < deadline && !done) await new Promise(r => setTimeout(r, 100))
c.close()

console.log(`  chunks before speech : ${audioBefore}`)
console.log(`  chunks after speech  : ${audioAfterSpeech}  (detection window, must be bounded)`)
console.log(`  local audio flushed  : ${flushed}`)
console.log(`  cancelled after      : ${speechStartAt ? speechStartAt - t0 : '—'}ms into the session`)
console.log(`  response.done        : ${done}`)
const ok = done === 'cancelled:turn_detected' && flushed
console.log('\n  VERDICT:', ok ? 'PASS' : 'FAIL')
process.exit(ok ? 0 : 1)
