/**
 * OpenCC Simplified -> Traditional on the user's transcript.
 *
 * Paraformer (the only Chinese STT the framework offers) emits Simplified, so a
 * zh-TW demo showed 帮我查一下今天那个台北的天气 in the user's own bubble while the
 * assistant replied in Traditional.
 *
 * CHARACTER-level conversion only (`to: 'tw'`), because a transcript must stay
 * invariant at the phonetic level -- it is a record of what was said. OpenCC's
 * S2T character mappings are homophonous, so the spoken form survives:
 *
 *   臺北 / 台北  both tái-běi
 *   餵 / 喂      both wèi
 *
 * which is why no character-level pick needs correcting here, even when it is
 * not the one a writer would have chosen.
 *
 * The `twp` preset must NOT be used *on the transcript*. It substitutes regional
 * vocabulary, which changes pronunciation and so changes the record: 用鼠标点击 becomes
 * 用滑鼠點選 (shǔbiāo diǎnjī -> huáshǔ diǎnxuǎn), a sentence the user never said.
 *
 * The assistant's own reply is the opposite case -- see toTWP() below.
 *
 * Conversion is display-only; the audio and the text the model receives are
 * untouched, and the raw STT output is kept in the debug export.
 */

// ~1 MB of dictionaries, loaded lazily and only in the cn -> t direction. Until
// the chunk arrives toTW() is the identity function, so the app renders
// immediately and nothing on the audio path waits for it.
let convert = null      // transcript: characters only
let convertP = null     // assistant reply: characters + Taiwanese vocabulary
const ready = import('opencc-js/cn2t')
  .then((OpenCC) => {
    convert = OpenCC.Converter({ from: 'cn', to: 'tw' })
    convertP = OpenCC.Converter({ from: 'cn', to: 'twp' })
  })
  .catch(() => {
    convert = null
    convertP = null
  })

export function toTW(text) {
  if (!text || !convert) return text
  try {
    return convert(text)
  } catch {
    return text // never let a display nicety break the transcript
  }
}

/**
 * Simplified -> Traditional with Taiwanese vocabulary, for the assistant's reply.
 *
 * `twp` rather than `tw` here, and deliberately so. The transcript is a record of
 * what a person said, so it must stay phonetically invariant; the reply is the
 * assistant's own words, so rendering them in Taiwanese usage (軟件 -> 軟體,
 * 鼠标点击 -> 滑鼠點選) is a correction rather than a falsification. The model still
 * answers in Simplified now and then despite the system prompt, and this is a
 * zh-TW demo.
 *
 * Display only: the audio was already synthesized from the text the model produced,
 * so a converted phrase can differ in wording from what was spoken. The raw text is
 * kept in the debug export.
 */
export function toTWP(text) {
  if (!text || !convertP) return text
  try {
    return convertP(text)
  } catch {
    return text
  }
}

/** Resolves once conversion is live; for tests, and to re-render early turns. */
export const zhtwReady = ready
