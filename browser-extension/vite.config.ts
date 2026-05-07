import { defineConfig } from "vite";
import { crx } from "@crxjs/vite-plugin";
import manifest from "./manifest.json";

// Bundle the extension via @crxjs/vite-plugin so manifest paths get
// rewritten to the dist/ output filenames + the dev-server reload
// machinery is wired up automatically when running `vite`.
export default defineConfig({
    plugins: [
        // @ts-ignore — manifest type narrowing
        crx({ manifest })
    ],
    build: {
        outDir: "dist",
        emptyOutDir: true,
        rollupOptions: {
            // Keep individual entries; the extension can't be a single
            // bundled file because MV3 enforces separate background
            // worker / popup / options scripts.
            input: {
                background: "src/background.ts",
                popup:      "src/popup.html",
                options:    "src/options.html"
            }
        }
    }
});
