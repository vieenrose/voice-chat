// One full VOICE turn driven through the frontend's own protocol client, so the
// UI's event handling is what gets tested and not merely the server's.
//
//   node checks/turn.mjs        (needs the pipeline up on :8765)
globalThis.location = { hostname: process.env.RT_HOST || '127.0.0.1' }
const { RealtimeClient } = await import(new URL('../src/lib/realtime.js', import.meta.url))

import fs from 'node:fs'
const wav = fs.readFileSync(new URL('../../asr_example.wav', import.meta.url))
const pcm = new Int16Array(wav.buffer, wav.byteOffset + 44, (wav.length - 44) >> 1)

const seen = []
let userPartials = 0, lastPartial = '', userFinal = '', assistant = '', audioChunks = 0, doneStatus = null
let firstAudioAt = 0
const t0 = Date.now()

const c = new RealtimeClient({
  onSpeechStart: () => seen.push('speech_start'),
  onSpeechStop:  () => seen.push('speech_stop'),
  onUserPartial: (t) => { userPartials++; lastPartial = t },
  onUserFinal:   (t) => { userFinal = t; seen.push('user_final') },
  onResponseStart: () => seen.push('response_start'),
  onAssistantText: (t) => { assistant += t },
  onAudio: () => { audioChunks++; if (!firstAudioAt) firstAudioAt = Date.now() - t0 },
  onResponseDone: (s, r) => { doneStatus = `${s}${r ? ':' + r : ''}`; seen.push('response_done') },
  onError: (m) => seen.push('ERROR:' + m),
})

await c.connect('ws://127.0.0.1:8765/v1/realtime', '你是一個親切的語音助理，請用繁體中文簡短回答。')

const step = 16000 * 0.16
for (let i = 0; i < pcm.length; i += step) {
  const slice = pcm.subarray(i, Math.min(i + step, pcm.length))
  c.appendAudio(Buffer.from(slice.buffer, slice.byteOffset, slice.byteLength).toString('base64'))
  await new Promise(r => setTimeout(r, 160))
}
const deadline = Date.now() + 45000
while (Date.now() < deadline && !seen.includes('response_done')) await new Promise(r => setTimeout(r, 200))
c.close()

console.log('  events        :', seen.join(' → '))
console.log('  user partials :', userPartials, '| last:', JSON.stringify(lastPartial))
console.log('  user final    :', JSON.stringify(userFinal))
console.log('  assistant     :', JSON.stringify(assistant.slice(0, 120)))
console.log('  audio chunks  :', audioChunks, '| first audio', firstAudioAt + 'ms')
console.log('  done status   :', doneStatus)

const ok = userFinal && assistant && audioChunks > 0 && doneStatus?.startsWith('completed')
// The cumulative-delta trap: partials must REPLACE. If they were appended, the
// last partial would be far longer than the final transcript.
const dupBug = lastPartial.length > userFinal.length * 1.5
console.log('\n  VERDICT:', ok && !dupBug ? 'PASS' : 'FAIL',
            dupBug ? '(cumulative partials were being appended)' : '')
process.exit(ok && !dupBug ? 0 : 1)
