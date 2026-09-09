import React from "react";
import { ArrowRight } from "lucide-react";
import { C } from "../theme.js";

export default function CausalStrip() {
  const steps = [
    { l: "Agent", c: C.blue },
    { l: "Failure", c: C.red },
    { l: "Weakness", c: C.yellow },
    { l: "Generated environment", c: C.orange },
    { l: "Improvement", c: C.green },
  ];
  return (
    <div className="flex items-center gap-2 px-5 py-2 overflow-x-auto" style={{ background: C.bg, borderBottom: `1px solid ${C.border}` }}>
      {steps.map((s, i) => (
        <React.Fragment key={s.l}>
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full whitespace-nowrap" style={{ background: `${s.c}18`, border: `1px solid ${s.c}44` }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.c }} />
            <span className="text-[11px] font-medium" style={{ color: s.c }}>{s.l}</span>
          </div>
          {i < steps.length - 1 && <ArrowRight size={12} color={C.textFaint} className="shrink-0" />}
        </React.Fragment>
      ))}
      <span className="text-[11px] ml-2" style={{ color: C.textFaint }}>
        — the AI trains itself by being placed in environments designed around its own weaknesses
      </span>
    </div>
  );
}
