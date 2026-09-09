import React from "react";
import { C } from "../theme.js";
import { Panel } from "../components/ui.jsx";
import { GeneralizationChart } from "../components/charts.jsx";
import { generalization } from "../lib/mockData.js";

export default function EvaluationPage() {
  return (
    <div className="p-5 flex flex-col gap-4">
      <div className="grid grid-cols-4 gap-4">
        {generalization.map((g) => (
          <Panel key={g.split} title={g.split}>
            <div className="text-[24px] font-semibold font-mono" style={{ color: g.split === "OOD" ? C.orange : C.text }}>{g.success}%</div>
            <div className="text-[11px] mt-1" style={{ color: C.textDim }}>success rate</div>
          </Panel>
        ))}
      </div>
      <Panel title="Success rate across evaluation splits">
        <GeneralizationChart />
      </Panel>
      <Panel pad={false} title="Split metrics">
        <table className="w-full text-[13px]">
          <thead>
            <tr style={{ color: C.textFaint, borderBottom: `1px solid ${C.border}` }}>
              {["Split", "Success rate", "Avg reward", "Avg steps", "Collision rate"].map((h) => (
                <th key={h} className="text-left font-normal px-4 py-2.5 text-[11px] uppercase tracking-wider">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {generalization.map((g) => (
              <tr key={g.split} style={{ borderBottom: `1px solid ${C.border}` }}>
                <td className="px-4 py-2.5" style={{ color: C.text }}>{g.split}</td>
                <td className="px-4 py-2.5 font-mono" style={{ color: C.textDim }}>{g.success}%</td>
                <td className="px-4 py-2.5 font-mono" style={{ color: C.textDim }}>{g.reward}</td>
                <td className="px-4 py-2.5 font-mono" style={{ color: C.textDim }}>{g.steps}</td>
                <td className="px-4 py-2.5 font-mono" style={{ color: g.collision > 10 ? C.red : C.textDim }}>{g.collision}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      <div className="text-[11px]" style={{ color: C.textFaint }}>
        Generalization gap (training → OOD): <span className="font-mono" style={{ color: C.orange }}>−30 pts</span>. The agent trains only on the training split; validation, test, and OOD environments are never seen during training.
      </div>
    </div>
  );
}
