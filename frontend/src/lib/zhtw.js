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
 * The `twp` preset must NOT be used. It substitutes regional vocabulary, which
 * changes pronunciation and so changes the transcript: 用鼠标点击 becomes
 * 用滑鼠點選 (shǔbiāo diǎnjī -> huáshǔ diǎnxuǎn), a sentence the user never said.
 *
 * Conversion is display-only; the audio and the text the model receives are
 * untouched, and the raw STT output is kept in the debug export.
 */

// ~1 MB of dictionaries, loaded lazily and only in the cn -> t direction. Until
// the chunk arrives toTW() is the identity function, so the app renders
// immediately and nothing on the audio path waits for it.
let convert = null
const ready = import('opencc-js/cn2t')
  .then((OpenCC) => {
    convert = OpenCC.Converter({ from: 'cn', to: 'tw' })
  })
  .catch(() => {
    convert = null
  })

export function toTW(text) {
  if (!text || !convert) return text
  try {
    return convert(text)
  } catch {
    return text // never let a display nicety break the transcript
  }
}

/** Resolves once conversion is live; for tests, and to re-render early turns. */
export const zhtwReady = ready
