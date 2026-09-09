import React, { useState } from "react";
import { Search, PlusCircle } from "lucide-react";
import { C } from "../theme.js";
import { Panel, StatusDot } from "../components/ui.jsx";
import { experiments } from "../lib/mockData.js";

export default function ExperimentsPage({ setPage }) {
  const [filter, setFilter] = useState("all");
  const filtered = experiments.filter((e) => filter === "all" || e.status === filter);
  return (
    <div className="p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {["all", "training", "completed", "failed"].map((f) => (
            <button key={f} onClick={() => setFilter(f)} className="px-3 py-1.5 rounded-md text-[12px] capitalize" style={{
              background: filter === f ? C.blueDim : "transparent",
              color: filter === f ? "#bcd6ff" : C.textDim,
              border: `1px solid ${filter === f ? C.blue : C.border}`,
            }}>{f}</button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md" style={{ border: `1px solid ${C.border}` }}>
            <Search size={13} color={C.textFaint} />
            <input placeholder="Search experiments" className="bg-transparent text-[12px] outline-none w-40" style={{ color: C.text }} />
          </div>
          <button onClick={() => setPage("create")} className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-medium" style={{ background: C.blue, color: "#04152e" }}>
            <PlusCircle size={13} /> New experiment
          </button>
        </div>
      </div>

      <Panel pad={false}>
        <table className="w-full text-[13px]">
          <thead>
            <tr style={{ color: C.textFaint, borderBottom: `1px solid ${C.border}` }}>
              {["Experiment", "Curriculum", "Algorithm", "Status", "Success rate", "Progress", "Created", ""].map((h) => (
                <th key={h} className="text-left font-normal px-4 py-2.5 text-[11px] uppercase tracking-wider">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((e) => (
              <tr key={e.id} style={{ borderBottom: `1px solid ${C.border}` }} className="hover:opacity-90">
                <td className="px-4 py-3 font-mono" style={{ color: C.text }}>{e.id}</td>
                <td className="px-4 py-3" style={{ color: C.textDim }}>{e.curriculum}</td>
                <td className="px-4 py-3" style={{ color: C.textDim }}>{e.algo}</td>
                <td className="px-4 py-3"><StatusDot status={e.status} /></td>
                <td className="px-4 py-3 font-mono" style={{ color: e.success > 0.6 ? C.green : e.success > 0.4 ? C.yellow : C.red }}>{Math.round(e.success * 100)}%</td>
                <td className="px-4 py-3 font-mono" style={{ color: C.textDim }}>{e.episodes}</td>
                <td className="px-4 py-3" style={{ color: C.textFaint }}>{e.created}</td>
                <td className="px-4 py-3">
                  <button onClick={() => setPage("details")} className="text-[11px]" style={{ color: C.blue }}>Open</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
