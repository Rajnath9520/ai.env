import React from "react";
import { C } from "../theme.js";
import { Panel, Bar1, Badge } from "../components/ui.jsx";
import { failureLog } from "../lib/mockData.js";

const details = [
  { key: "long_horizon_planning", title: "Long-horizon planning", severity: 0.78, confidence: 0.86, evidence: ["Decision points failed: 5", "Average planning horizon: 3.1", "Successful long routes: 28%"] },
  { key: "dead_end_recovery", title: "Dead-end recovery", severity: 0.64, confidence: 0.74, evidence: ["Dead-end entries: 4", "Successful recoveries: 1", "Average recovery steps: 17"] },
  { key: "path_efficiency", title: "Path efficiency", severity: 0.51, confidence: 0.71, evidence: ["Optimal path ratio: 0.61", "Redundant steps: 12", "Repeated states: 6"] },
  { key: "exploration", title: "Exploration", severity: 0.32, confidence: 0.6, evidence: ["Unique cells visited: 54%", "Time to first goal sighting: 41 steps"] },
];

export default function FailuresPage() {
  return (
    <div className="p-5 grid grid-cols-3 gap-4">
      <div className="col-span-2 flex flex-col gap-3">
        {details.map((d) => (
          <Panel key={d.key} title={d.title} right={<Badge color={d.severity > 0.7 ? C.red : d.severity > 0.4 ? C.yellow : C.green}>severity {d.severity.toFixed(2)}</Badge>}>
            <div className="grid grid-cols-2 gap-4 mb-3">
              <div>
                <div className="flex justify-between text-[11px] mb-1" style={{ color: C.textDim }}><span>Severity</span><span className="font-mono">{d.severity.toFixed(2)}</span></div>
                <Bar1 value={d.severity} color={C.red} />
              </div>
              <div>
                <div className="flex justify-between text-[11px] mb-1" style={{ color: C.textDim }}><span>Confidence</span><span className="font-mono">{d.confidence.toFixed(2)}</span></div>
                <Bar1 value={d.confidence} color={C.blue} />
              </div>
            </div>
            <div className="flex flex-col gap-1">
              {d.evidence.map((e) => (
                <div key={e} className="text-[12px] flex items-center gap-2" style={{ color: C.textDim }}>
                  <span className="w-1 h-1 rounded-full" style={{ background: C.textFaint }} />{e}
                </div>
              ))}
            </div>
          </Panel>
        ))}
      </div>
      <Panel title="Episode failure log">
        <table className="w-full text-[12px]">
          <thead>
            <tr style={{ color: C.textFaint }}>
              <th className="text-left font-normal pb-2">Ep</th>
              <th className="text-left font-normal pb-2">Type</th>
              <th className="text-right font-normal pb-2">Sev.</th>
              <th className="text-right font-normal pb-2">Conf.</th>
            </tr>
          </thead>
          <tbody>
            {failureLog.map((f) => (
              <tr key={f.ep} style={{ borderTop: `1px solid ${C.border}` }}>
                <td className="py-1.5 font-mono" style={{ color: C.textDim }}>{f.ep}</td>
                <td className="py-1.5" style={{ color: C.text }}>{f.type}</td>
                <td className="py-1.5 text-right font-mono" style={{ color: f.severity > 0.7 ? C.red : C.yellow }}>{f.severity.toFixed(2)}</td>
                <td className="py-1.5 text-right font-mono" style={{ color: C.textDim }}>{f.confidence.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
