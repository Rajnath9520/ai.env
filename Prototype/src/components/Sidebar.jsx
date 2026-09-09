import React from "react";
import {
  LayoutDashboard, FlaskConical, PlusCircle, Activity, LayoutGrid,
  AlertTriangle, GitBranch, Target, BarChart3, FileText, Cpu, HardDrive,
  Server, Thermometer, Clock, Zap,
} from "lucide-react";
import { C } from "../theme.js";
import { Panel, Bar1 } from "./ui.jsx";

export const NAV = [
  { id: "overview", label: "Lab overview", icon: LayoutDashboard },
  { id: "experiments", label: "Experiments", icon: FlaskConical },
  { id: "create", label: "New experiment", icon: PlusCircle },
  { id: "live", label: "Live training", icon: Activity },
  { id: "environments", label: "Environments", icon: LayoutGrid },
  { id: "failures", label: "Failure analysis", icon: AlertTriangle },
  { id: "curriculum", label: "Curriculum", icon: GitBranch },
  { id: "evaluation", label: "Evaluation", icon: Target },
  { id: "comparison", label: "Comparison", icon: BarChart3 },
  { id: "details", label: "Experiment details", icon: FileText },
];

export default function Sidebar({ page, setPage }) {
  return (
    <div className="w-[220px] shrink-0 flex flex-col" style={{ background: C.sidebar, borderRight: `1px solid ${C.border}` }}>
      <div className="px-4 py-4" style={{ borderBottom: `1px solid ${C.border}` }}>
        <div className="text-[13px] font-bold tracking-widest" style={{ color: C.text }}> ai.env </div>
        <div className="text-[10px] tracking-wider mt-0.5" style={{ color: C.textFaint }}>ENVIRONMENT LAB</div>
      </div>
      <div className="flex-1 py-3 px-2 flex flex-col gap-0.5">
        {NAV.map((n) => {
          const Icon = n.icon;
          const active = page === n.id;
          return (
            <button
              key={n.id}
              onClick={() => setPage(n.id)}
              className="flex items-center gap-2.5 px-3 py-2 rounded-md text-[13px] text-left transition-colors"
              style={{ background: active ? C.blueDim : "transparent", color: active ? "#bcd6ff" : C.textDim }}
            >
              <Icon size={15} />
              {n.label}
            </button>
          );
        })}
      </div>
      <div className="p-3">
        <Panel title="System status" pad={true}>
          <div className="flex flex-col gap-2.5">
            {[
              { l: "GPU", v: 67, icon: Zap },
              { l: "CPU", v: 41, icon: Cpu },
              { l: "Memory", v: 63, icon: Server },
              { l: "Disk", v: 72, icon: HardDrive },
            ].map((s) => (
              <div key={s.l}>
                <div className="flex items-center justify-between text-[11px] mb-1" style={{ color: C.textDim }}>
                  <span className="flex items-center gap-1.5"><s.icon size={11} />{s.l}</span>
                  <span className="font-mono">{s.v}%</span>
                </div>
                <Bar1 value={s.v / 100} color={s.v > 65 ? C.orange : C.blue} height={4} />
              </div>
            ))}
            <div className="flex items-center justify-between text-[11px] pt-1" style={{ color: C.textFaint, borderTop: `1px solid ${C.border}` }}>
              <span className="flex items-center gap-1.5"><Thermometer size={11} />GPU temp</span>
              <span className="font-mono">62°C</span>
            </div>
            <div className="flex items-center justify-between text-[11px]" style={{ color: C.textFaint }}>
              <span className="flex items-center gap-1.5"><Clock size={11} />Uptime</span>
              <span className="font-mono">2d 04:21</span>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
