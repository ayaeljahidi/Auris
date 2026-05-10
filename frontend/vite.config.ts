import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    port: 3000,
    // Proxy all API calls and WebSocket connections to the FastAPI backend.
    // This means the frontend can use relative paths (/transcribe, /ws/live)
    // instead of hardcoded http://localhost:8000 URLs.
    proxy: {
      '/transcribe': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/emotion': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws/live': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
