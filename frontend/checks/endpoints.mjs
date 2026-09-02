// URL building for the added HTTP routes. A path carrying a query must survive:
// assigning it to URL.pathname encodes the '?', which silently 404'd the model
// catalogue and showed an empty dropdown with no error.
//
//   node checks/endpoints.mjs        (no server needed)

const { serverUrl } = await import(new URL('../src/lib/endpoints.js', import.meta.url))

const cases = [
  ['ws://h:8765/v1/realtime',  '/v1/vram',                        'http://h:8765/v1/vram'],
  ['ws://h:8765/v1/realtime',  '/v1/llm-models?provider=openrouter',
                               'http://h:8765/v1/llm-models?provider=openrouter'],
  ['wss://h.ts.net:8443/v1/realtime', '/v1/llm-config',           'https://h.ts.net:8443/v1/llm-config'],
  ['wss://h.ts.net:8443/v1/realtime', '/v1/llm-models?provider=openrouter',
                               'https://h.ts.net:8443/v1/llm-models?provider=openrouter'],
]
let bad = 0
for (const [ep, path, want] of cases) {
  const got = serverUrl(ep, path)
  const ok = got === want
  if (!ok) bad++
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${path} -> ${got}${ok ? '' : `  (want ${want})`}`)
}
// the specific regression: no percent-encoded '?'
const enc = serverUrl('ws://h:8765/v1/realtime', '/v1/llm-models?provider=openrouter')
if (enc.includes('%3F')) { console.log('  FAIL query was percent-encoded'); bad++ }
if (serverUrl('not a url', '/v1/vram') !== null) { console.log('  FAIL bad endpoint should yield null'); bad++ }
console.log('\n  VERDICT:', bad ? 'FAIL' : 'PASS')
process.exit(bad ? 1 : 0)
