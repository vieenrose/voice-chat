import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: true,
    hmr: { host: "localhost" },
    proxy: {
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      },
      '/api': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/stats': 'http://localhost:8000'
    }
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
