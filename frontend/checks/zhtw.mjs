// The transcript converter: Traditional output, and invariant at the phonetic
// level, which is what a record of speech has to be.
//
//   node checks/zhtw.mjs        (no server needed)

const { toTW, toTWP, zhtwReady } = await import(new URL('../src/lib/zhtw.js', import.meta.url))
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

// The assistant's reply takes the opposite choice: twp, because these are the
// assistant's own words rather than a record of what someone said, so Taiwanese
// vocabulary is a correction and not a falsification.
console.log('  -- reply (toTWP) --')
const replyCases = [
  ['今天台北天气白天晴时多云，气温大约 26–33°C。', '今天台北天氣白天晴時多雲，氣溫大約 26–33°C。'],
  ['你好！很高兴见到你。有什么我可以帮你的吗？', '你好！很高興見到你。有什麼我可以幫你的嗎？'],
  ['这个软件用鼠标点击即可。', '這個軟體用滑鼠點選即可。'],   // vocabulary, unlike the transcript
]
for (const [input, want] of replyCases) {
  const got = toTWP(input)
  const ok = got === want
  if (!ok) bad++
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${input} -> ${got}${ok ? '' : `  (want ${want})`}`)
}
// the two converters must differ exactly where vocabulary differs
if (toTW('这个软件') === toTWP('这个软件')) {
  console.log('  FAIL transcript and reply converters should differ on vocabulary'); bad++
}
if (/[这软气帮吗]/.test(toTWP('这个软件很好，气温帮你查吗'))) {
  console.log('  FAIL simplified characters remain in the reply'); bad++
}
for (const v of ['', null, undefined]) {
  if (toTWP(v) !== v) { console.log('  FAIL empty reply input not preserved'); bad++ }
}

console.log('\n  VERDICT:', bad ? 'FAIL' : 'PASS')
process.exit(bad ? 1 : 0)
