import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        app: {
          bg: "#0a0e18",
          panel: "#0f1524",
          panel2: "#131a2c",
          border: "#1f2a40",
          hi: "#e6ebf5",
          mid: "#8b96ab",
          low: "#566078",
          blue: "#3f8cff",
          green: "#22c55e",
          orange: "#f97a2c",
          red: "#ef4444",
        },
      },
    },
  },
  plugins: [],
};
export default config;
