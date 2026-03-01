import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",          // IMPORTANT for Streamlit component hosting
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true
  }
});
