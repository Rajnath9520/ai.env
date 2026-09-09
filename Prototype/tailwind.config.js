/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        sans: ["ui-sans-serif", "system-ui", "Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};
