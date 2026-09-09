import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Lab -- Failure-Driven Environment Synthesis",
  description: "Experiment-control dashboard for the adaptive AI training research platform.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-app-bg text-app-hi text-sm" style={{ fontFamily: "Inter, ui-sans-serif, system-ui" }}>
        {children}
      </body>
    </html>
  );
}
