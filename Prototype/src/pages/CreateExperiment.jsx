import React, { useState } from "react";
import { CheckCircle2, Play } from "lucide-react";
import { C } from "../theme.js";
import { Panel, SectionLabel } from "../components/ui.jsx";

export default function CreateExperimentPage({ setPage }) {
  const [curriculum, setCurriculum] = useState("failure");
  const curricula = [
    { id: "random", label: "Random", desc: "Environments generated with fully random parameters within valid bounds." },
    { id: "fixed", label: "Fixed curriculum", desc: "Difficulty progresses through predefined easy → very hard stages." },
    { id: "generic", label: "Generic adaptive", desc: "Difficulty rises or falls based on overall success rate only." },
    { id: "failure", label: "Failure driven", desc: "Environments are synthesized to target the agent's measured behavioral weaknesses." },
  ];
  const inputStyle = { background: C.panelAlt, border: `1px solid ${C.border}`, color: C.text };
  return (
    <div className="p-5 flex flex-col gap-4 max-w-4xl">
      <Panel title="Configuration">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <SectionLabel>Algorithm</SectionLabel>
            <select className="w-full px-3 py-2 rounded-md text-[13px]" style={inputStyle}>
              <option>PPO (Stable-Baselines3)</option>
            </select>
          </div>
          <div>
            <SectionLabel>Environment type</SectionLabel>
            <select className="w-full px-3 py-2 rounded-md text-[13px]" style={inputStyle}>
              <option>GridWorld — 2D</option>
            </select>
          </div>
          <div>
            <SectionLabel>Grid size</SectionLabel>
            <input defaultValue="12 × 12" className="w-full px-3 py-2 rounded-md text-[13px] font-mono" style={inputStyle} />
          </div>
          <div>
            <SectionLabel>Random seed</SectionLabel>
            <input defaultValue="48291" className="w-full px-3 py-2 rounded-md text-[13px] font-mono" style={inputStyle} />
          </div>
          <div>
            <SectionLabel>Training episodes</SectionLabel>
            <input defaultValue="10,000" className="w-full px-3 py-2 rounded-md text-[13px] font-mono" style={inputStyle} />
          </div>
          <div>
            <SectionLabel>Evaluation frequency</SectionLabel>
            <input defaultValue="every 500 episodes" className="w-full px-3 py-2 rounded-md text-[13px] font-mono" style={inputStyle} />
          </div>
        </div>
      </Panel>

      <Panel title="Curriculum strategy">
        <div className="grid grid-cols-2 gap-3">
          {curricula.map((c) => (
            <button key={c.id} onClick={() => setCurriculum(c.id)} className="text-left p-3 rounded-md" style={{
              background: curriculum === c.id ? C.blueDim : C.panelAlt,
              border: `1px solid ${curriculum === c.id ? C.blue : C.border}`,
            }}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[13px] font-medium" style={{ color: curriculum === c.id ? "#bcd6ff" : C.text }}>{c.label}</span>
                {curriculum === c.id && <CheckCircle2 size={14} color={C.blue} />}
              </div>
              <div className="text-[12px]" style={{ color: C.textDim }}>{c.desc}</div>
            </button>
          ))}
        </div>
      </Panel>

      <div className="flex items-center justify-between">
        <div className="text-[11px]" style={{ color: C.textFaint }}>Experiment will be fully reproducible from this configuration and its random seed.</div>
        <button onClick={() => setPage("live")} className="flex items-center gap-1.5 px-4 py-2 rounded-md text-[13px] font-medium" style={{ background: C.blue, color: "#04152e" }}>
          <Play size={14} /> Launch experiment
        </button>
      </div>
    </div>
  );
}
