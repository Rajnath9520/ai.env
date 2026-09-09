import React, { useState } from "react";
import { C } from "../theme.js";
import { Panel, KV } from "../components/ui.jsx";
import { ComparisonChart, SuccessProgressChart } from "../components/charts.jsx";
import { learningProgressData } from "../lib/mockData.js";

const strategies = [
  { key: "Random", color: C.textFaint, ood: 41, episodesTo80: "—" },
  { key: "Fixed", color: C.yellow, ood: 54, episodesTo80: "8,400" },
  { key: "Generic adaptive", color: C.purple, ood: 61, episodesTo80: "6,150" },
  { key: "Failure driven", color: C.orange, ood: 76, episodesTo80: "3,900" },
];

export default function ComparisonPage() {
  const [active, setActive] = useState({ Random: true, Fixed: true, "Generic adaptive": true, "Failure driven": true });
  return (
    <div className="p-5 flex flex-col gap-4">
      <Panel title="Success rate vs episodes" right={
        <div className="flex items-center gap-2">
          {strategies.map((s) => (
            <button key={s.key} onClick={() => setActive((a) => ({ ...a, [s.key]: !a[s.key] }))} className="flex items-center gap-1.5 text-[11px] px-2 py-1 rounded" style={{ opacity: active[s.key] ? 1 : 0.4, color: s.color }}>
              <span className="w-2 h-2 rounded-full" style={{ background: s.color }} />{s.key}
            </button>
          ))}
        </div>
      }>
        <ComparisonChart />
      </Panel>

      <div className="grid grid-cols-4 gap-4">
        {strategies.map((s) => (
          <Panel key={s.key} title={s.key}>
            <KV label="OOD generalization" value={`${s.ood}%`} valueColor={s.color} />
            <div className="mt-3"><KV label="Episodes to 80% success" value={s.episodesTo80} /></div>
          </Panel>
        ))}
      </div>

      <Panel title="Learning progress by curriculum stage">
        <SuccessProgressChart data={learningProgressData} />
      </Panel>
    </div>
  );
}
