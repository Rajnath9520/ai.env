import React, { useMemo } from "react";
import { ChevronRight, CheckCircle2, Play, Circle, Activity, AlertTriangle, LayoutGrid, BarChart3 } from "lucide-react";
import { C } from "../theme.js";
import { Panel, Bar1, KV } from "../components/ui.jsx";
import { GridWorld, MiniGrid } from "../components/GridWorld.jsx";
import { SuccessProgressChart } from "../components/charts.jsx";
import { makeGrid, weaknesses, failureLog, curriculumEnvs, learningProgressData } from "../lib/mockData.js";

export default function OverviewPage({ setPage }) {
  const bigGrid = useMemo(() => makeGrid(48291, 12, 0.28), []);
  return (
    <div className="p-5 flex flex-col gap-4">
      <div className="grid grid-cols-3 gap-4">
        <Panel title="Live environment · GridWorld-042 · 12×12" className="col-span-2">
          <GridWorld grid={bigGrid} cell={26} />
          <div className="grid grid-cols-4 gap-3 mt-4 pt-3" style={{ borderTop: `1px solid ${C.border}` }}>
            <KV label="Obstacle density" value="0.38" />
            <KV label="Path complexity" value="0.72" />
            <KV label="Dead-end rate" value="0.44" />
            <KV label="Difficulty" value="0.61" />
          </div>
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="Agent status">
            <div className="grid grid-cols-2 gap-3">
              <KV label="Success rate" value="74.2%" valueColor={C.green} />
              <KV label="Avg reward" value="68.4" />
              <KV label="Avg steps" value="31.7" />
              <KV label="Collision rate" value="8.3%" valueColor={C.red} />
            </div>
          </Panel>
          <Panel title="Current weakness">
            <div className="text-[13px] font-semibold mb-2" style={{ color: C.orange }}>Long-horizon planning</div>
            <div className="flex flex-col gap-2">
              {weaknesses.map((w) => (
                <div key={w.key}>
                  <div className="flex justify-between text-[11px] mb-1" style={{ color: C.textDim }}>
                    <span>{w.label}</span><span className="font-mono">{w.value.toFixed(2)}</span>
                  </div>
                  <Bar1 value={w.value} color={w.color} />
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Panel title="Failure analysis — recent episodes" right={<button onClick={() => setPage("failures")} className="text-[11px] flex items-center gap-1" style={{ color: C.blue }}>View all<ChevronRight size={12} /></button>}>
          <table className="w-full text-[12px]">
            <thead>
              <tr style={{ color: C.textFaint }}>
                <th className="text-left font-normal pb-2">Episode</th>
                <th className="text-left font-normal pb-2">Type</th>
                <th className="text-right font-normal pb-2">Severity</th>
              </tr>
            </thead>
            <tbody>
              {failureLog.slice(0, 4).map((f) => (
                <tr key={f.ep} style={{ borderTop: `1px solid ${C.border}` }}>
                  <td className="py-1.5 font-mono" style={{ color: C.textDim }}>{f.ep}</td>
                  <td className="py-1.5" style={{ color: C.text }}>{f.type}</td>
                  <td className="py-1.5 text-right font-mono" style={{ color: f.severity > 0.7 ? C.red : C.yellow }}>{f.severity.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel title="Curriculum timeline" right={<button onClick={() => setPage("curriculum")} className="text-[11px] flex items-center gap-1" style={{ color: C.blue }}>View all<ChevronRight size={12} /></button>}>
          <div className="flex items-center gap-1 overflow-x-auto pb-1">
            {curriculumEnvs.map((e, i) => (
              <React.Fragment key={e.id}>
                <div className="flex flex-col items-center gap-1 shrink-0">
                  <div className="w-7 h-7 rounded-full flex items-center justify-center" style={{
                    background: e.status === "done" ? C.greenDim : e.status === "current" ? C.blueDim : C.panelAlt,
                    border: `1px solid ${e.status === "done" ? C.green : e.status === "current" ? C.blue : C.border}`,
                  }}>
                    {e.status === "done" ? <CheckCircle2 size={13} color={C.green} /> : e.status === "current" ? <Play size={11} color={C.blue} /> : <Circle size={10} color={C.textFaint} />}
                  </div>
                  <span className="text-[9px] font-mono" style={{ color: C.textFaint }}>{e.id.split("-")[1]}</span>
                </div>
                {i < curriculumEnvs.length - 1 && <div className="w-3 h-px shrink-0" style={{ background: C.border }} />}
              </React.Fragment>
            ))}
          </div>
        </Panel>

        <Panel title="Learning progress">
          <SuccessProgressChart data={learningProgressData} height={140} />
        </Panel>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {[
          { id: "live", label: "Live training", icon: Activity, body: <div className="mt-2"><MiniGrid seed={91} cell={7} /></div> },
          { id: "failures", label: "Failure analysis", icon: AlertTriangle, body: <div className="flex flex-col gap-1.5 mt-2">{weaknesses.slice(0, 3).map((w) => <Bar1 key={w.key} value={w.value} color={w.color} height={4} />)}</div> },
          { id: "environments", label: "Environment generator", icon: LayoutGrid, body: <div className="flex gap-1.5 mt-2">{[201, 202, 203].map((s) => <MiniGrid key={s} seed={s} cell={4} />)}</div> },
          { id: "comparison", label: "Experiment comparison", icon: BarChart3, body: <div className="mt-2"><SuccessProgressChart data={learningProgressData.slice(0, 12)} height={70} /></div> },
        ].map((card) => (
          <button key={card.id} onClick={() => setPage(card.id)} className="text-left">
            <Panel title={card.label} right={<card.icon size={13} color={C.textFaint} />}>{card.body}</Panel>
          </button>
        ))}
      </div>
    </div>
  );
}
