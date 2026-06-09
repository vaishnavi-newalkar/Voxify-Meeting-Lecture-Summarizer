// import { defineConfig, loadEnv } from 'vite';
// import react from '@vitejs/plugin-react';

// // https://vite.dev/config/
// export default defineConfig({
//   plugins: [react()],
//   define: {
//     'import.meta.env.VITE_API_URL': JSON.stringify(import.meta.env.VITE_API_URL || ''),
//   },
//   test: {
//     globals: true,
//     environment: 'jsdom',
//     setupFiles: ['./src/__tests__/setup.js'],
//   },
// });
/* global process */
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react()],
    define: {
      'import.meta.env.VITE_API_URL': JSON.stringify(
        env.VITE_API_URL || ''
      ),
    },
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./src/__tests__/setup.js'],
    },
  };
});
