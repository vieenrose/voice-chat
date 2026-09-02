// The debug-log export: does it capture what a bug report needs, without the
// audio payloads that would make the file unusable?
//
//   node checks/log.mjs        (no server needed -- synthetic frames)

globalThis.location = { hostname: 'h', protocol: 'https:' }
globalThis.performance ??= { now: () => Date.now() }
globalThis.WebSocket = { OPEN: 1 }
const { RealtimeClient } = await import(new URL('../src/lib/realtime.js', import.meta.url))

const c = new RealtimeClient({})
c.t0 = performance.now()
c.ws = { readyState: 1, send() {} }

const AUDIO_CHUNKS = 300
const B64 = 12288
for (let i = 0; i < AUDIO_CHUNKS; i++) {
  c._record('in', { type: 'response.output_audio.delta', delta: 'x'.repeat(B64) })
}
c._record('in', {
  type: 'conversation.item.input_audio_transcription.completed',
  transcript: '欢迎大家来体验',
})
c._record('in', {
  type: 'response.done',
  response: { status: 'cancelled', status_details: { reason: 'turn_detected' } },
})
c.sendText('台北？')
for (let i = 0; i < 50; i++) c.appendAudio('m'.repeat(5120))

const log = c.log
const elided = log.filter((e) => typeof e.delta === 'string' && e.delta.includes('elided'))
const appends = log.filter((e) => e.type === 'input_audio_buffer.append')
const bytes = JSON.stringify(log, null, 2).length

const checks = [
  ['every audio delta elided', elided.length === AUDIO_CHUNKS],
  ['mic frames counted, not listed', appends.length === 0 && c.micFrames === 50],
  ['outbound frames recorded', log.some((e) => e.dir === 'out' && e.type === 'response.create')],
  ['cancel reason preserved', log.find((e) => e.type === 'response.done')?.response
    ?.status_details?.reason === 'turn_detected'],
  ['transcripts preserved', log.some((e) => e.transcript === '欢迎大家来体验')],
  ['export stays small', bytes < 200 * 1024],
  ['ring buffer is bounded', c.logLimit > 0 && c.logLimit <= 10000],
]
for (const [what, ok] of checks) console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${what}`)
console.log(`\n  ${(bytes / 1024).toFixed(1)} KB exported, vs ${(AUDIO_CHUNKS * B64 / 1048576).toFixed(1)} MB of raw audio avoided`)
const pass = checks.every(([, ok]) => ok)
console.log('  VERDICT:', pass ? 'PASS' : 'FAIL')
process.exit(pass ? 0 : 1)
