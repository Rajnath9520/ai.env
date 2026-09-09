import React from "react";
import { CheckCircle2, Play, Circle } from "lucide-react";
import { C } from "../theme.js";
import { Panel, StatusDot } from "../components/ui.jsx";
import { curriculumEnvs } from "../lib/mockData.js";

export default function CurriculumPage() {
  return (
    <div className="p-5 flex flex-col gap-4">
      <Panel title="Curriculum timeline">
        <div className="flex items-center gap-1 overflow-x-auto pb-2">
          {curriculumEnvs.map((e, i) => (
            <React.Fragment key={e.id}>
              <div className="flex flex-col items-center gap-1.5 shrink-0 px-2">
                <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{
                  background: e.status === "done" ? C.greenDim : e.status === "current" ? C.blueDim : C.panelAlt,
                  border: `1px solid ${e.status === "done" ? C.green : e.status === "current" ? C.blue : C.border}`,
                }}>
                  {e.status === "done" ? <CheckCircle2 size={16} color={C.green} /> : e.status === "current" ? <Play size={14} color={C.blue} /> : <Circle size={12} color={C.textFaint} />}
                </div>
                <span className="text-[10px] font-mono" style={{ color: C.textDim }}>{e.id}</span>
                <span className="text-[9px]" style={{ color: C.textFaint }}>{e.difficulty.toFixed(2)}</span>
              </div>
              {i < curriculumEnvs.length - 1 && <div className="w-6 h-px shrink-0" style={{ background: C.border }} />}
            </React.Fragment>
          ))}
        </div>
      </Panel>

      <Panel pad={false} title="Environment history">
        <table className="w-full text-[13px]">
          <thead>
            <tr style={{ color: C.textFaint, borderBottom: `1px solid ${C.border}` }}>
              {["Env ID", "Target weakness", "Difficulty", "Success rate", "Learning progress", "Status"].map((h) => (
                <th key={h} className="text-left font-normal px-4 py-2.5 text-[11px] uppercase tracking-wider">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {curriculumEnvs.map((e) => (
              <tr key={e.id} style={{ borderBottom: `1px solid ${C.border}`, background: e.status === "current" ? C.blueDim : "transparent" }}>
                <td className="px-4 py-2.5 font-mono" style={{ color: C.text }}>{e.id}</td>
                <td className="px-4 py-2.5" style={{ color: C.textDim }}>{e.weakness}</td>
                <td className="px-4 py-2.5 font-mono" style={{ color: C.textDim }}>{e.difficulty.toFixed(2)}</td>
                <td className="px-4 py-2.5 font-mono" style={{ color: e.success ? C.green : C.textFaint }}>{e.success ? `${Math.round(e.success * 100)}%` : "—"}</td>
                <td className="px-4 py-2.5 font-mono" style={{ color: e.lp ? C.green : C.textFaint }}>{e.lp ? `+${e.lp.toFixed(2)}` : "—"}</td>
                <td className="px-4 py-2.5"><StatusDot status={e.status === "done" ? "completed" : e.status === "current" ? "training" : "pending"} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
