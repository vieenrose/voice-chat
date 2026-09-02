import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: true,
    hmr: { host: "localhost" },
    // No proxy. The app talks the OpenAI Realtime protocol straight to the
    // speech-to-speech server on :8765 (endpoint is editable in the UI); it uses
    // no HTTP API of its own. The old proxy pointed at app.py on :8000, which
    // this frontend no longer speaks to.
  },
  preview: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: true
  },
  build: {
    target: 'esnext'
  }
})
