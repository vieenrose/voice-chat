// The transcript converter: Traditional output, and invariant at the phonetic
// level, which is what a record of speech has to be.
//
//   node checks/zhtw.mjs        (no server needed)

const { toTW, zhtwReady } = await import(new URL('../src/lib/zhtw.js', import.meta.url))
await zhtwReady

const cases = [
  // [input, expected] -- straight OpenCC S2T, no hand corrections
  ['帮我查一下今天那个台北的天气', '幫我查一下今天那個臺北的天氣'],
  ['那请问一加一等于多少', '那請問一加一等於多少'],
  ['好谢谢', '好謝謝'],
  ['这个软件不错', '這個軟件不錯'],  // NOT 軟體: that would change the spoken form
  ['用鼠标点击', '用鼠標點擊'],        // NOT 滑鼠點選, same reason
]
let bad = 0
for (const [input, want] of cases) {
  const got = toTW(input)
  const ok = got === want
  if (!ok) bad++
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${input} -> ${got}${ok ? '' : `  (want ${want})`}`)
}
// Nothing may be left Simplified, and empty input must survive.
const leftover = toTW('帮我查这个软件的天气').match(/[帮这软气个]/)
if (leftover) { console.log('  FAIL simplified characters remain:', leftover[0]); bad++ }
for (const v of ['', null, undefined]) {
  if (toTW(v) !== v) { console.log('  FAIL empty input not preserved:', JSON.stringify(v)); bad++ }
}
console.log('\n  VERDICT:', bad ? 'FAIL' : 'PASS')
process.exit(bad ? 1 : 0)
