/**
 * Build a server URL from the Realtime endpoint the user is connected to.
 *
 * Extracted and tested because getting it wrong is silent: assigning a path that
 * already contains a query to `URL.pathname` percent-encodes the `?`, so
 * `/v1/llm-models?provider=openrouter` became `/v1/llm-models%3Fprovider=openrouter`,
 * the server answered 404, and the UI showed an empty model list — 模型（0/0） — with
 * no error anywhere.
 */
export function serverUrl(wsEndpoint, path) {
  try {
    const u = new URL(String(wsEndpoint).replace(/^ws/, 'http'))
    const [pathname, search = ''] = String(path).split('?')
    u.pathname = pathname
    u.search = search
    u.hash = ''
    return u.toString()
  } catch {
    return null
  }
}
