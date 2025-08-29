import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Use /app/ when served behind FastAPI at /app
  base: process.env.VITE_BASE || '/',
  server: { port: 5173, strictPort: true },
  build: { outDir: 'dist', sourcemap: true }
})
