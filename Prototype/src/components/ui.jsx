import React from "react";
import { C } from "../theme.js";

export function Panel({ title, right, children, className = "", pad = true }) {
  return (
    <div className={`rounded-lg ${className}`} style={{ background: C.panel, border: `1px solid ${C.border}` }}>
      {title && (
        <div className="flex items-center justify-between px-4 pt-3 pb-2" style={{ borderBottom: `1px solid ${C.border}` }}>
          <span className="text-[11px] font-semibold tracking-widest uppercase" style={{ color: C.textDim }}>{title}</span>
          {right}
        </div>
      )}
      <div className={pad ? "p-4" : ""}>{children}</div>
    </div>
  );
}

export function Bar1({ value, color = C.blue, height = 6, track = C.panelAlt }) {
  return (
    <div className="w-full rounded-full overflow-hidden" style={{ height, background: track }}>
      <div style={{ width: `${Math.round(value * 100)}%`, height: "100%", background: color, borderRadius: 999 }} />
    </div>
  );
}

export function Badge({ children, color = C.blue, bg }) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium" style={{ color, background: bg || `${color}22`, border: `1px solid ${color}55` }}>
      {children}
    </span>
  );
}

export function StatusDot({ status }) {
  const map = { training: C.blue, completed: C.green, failed: C.red, pending: C.textFaint };
  const color = map[status] || C.textFaint;
  return (
    <span className="inline-flex items-center gap-1.5 text-[12px]" style={{ color }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {status}
    </span>
  );
}

export function KV({ label, value, valueColor }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color: C.textFaint }}>{label}</div>
      <div className="text-[13px] font-mono" style={{ color: valueColor || C.text }}>{value}</div>
    </div>
  );
}

export function SectionLabel({ children }) {
  return <div className="text-[11px] font-semibold tracking-widest uppercase mb-2" style={{ color: C.textDim }}>{children}</div>;
}
