import React from "react";
import { Pause, Square } from "lucide-react";
import { C } from "../theme.js";
import { KV, StatusDot } from "./ui.jsx";

const titleMap = {
  overview: "Lab overview", experiments: "Experiments", create: "New experiment",
  live: "Live training", environments: "Environments", failures: "Failure analysis",
  curriculum: "Curriculum", evaluation: "Evaluation", comparison: "Comparison", details: "Experiment details",
};

export default function TopBar({ page }) {
  return (
    <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: `1px solid ${C.border}`, background: C.panelAlt }}>
      <div>
        <div className="text-[15px] font-semibold" style={{ color: C.text }}>{titleMap[page]}</div>
        <div className="text-[11px] mt-0.5" style={{ color: C.textFaint }}>Failure-driven environment synthesis for adaptive AI training</div>
      </div>
      <div className="flex items-center gap-5">
        <KV label="Experiment" value="EXP-024" />
        <div className="flex items-center gap-1.5"><StatusDot status="training" /></div>
        <KV label="Episode" value="1,842 / 10,000" />
        <KV label="Seed" value="48291" />
        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-medium" style={{ background: C.orangeDim, color: C.orange, border: `1px solid ${C.orange}55` }}>
          <Pause size={13} /> Pause
        </button>
        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-medium" style={{ background: C.redDim, color: C.red, border: `1px solid ${C.red}55` }}>
          <Square size={13} /> Stop
        </button>
      </div>
    </div>
  );
}
