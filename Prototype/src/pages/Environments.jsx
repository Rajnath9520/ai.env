import React, { useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { C } from "../theme.js";
import { Panel, Bar1, Badge, KV } from "../components/ui.jsx";
import { MiniGrid } from "../components/GridWorld.jsx";
import { candidates } from "../lib/mockData.js";

export default function EnvironmentsPage() {
  const [selected, setSelected] = useState("ENV-043");
  const sel = candidates.find((c) => c.id === selected);
  return (
    <div className="p-5 grid grid-cols-3 gap-4">
      <Panel title="Candidate environments — targeting long-horizon planning" className="col-span-2">
        <div className="grid grid-cols-2 gap-3">
          {candidates.map((c) => (
            <button key={c.id} onClick={() => setSelected(c.id)} className="text-left p-3 rounded-md" style={{
              background: selected === c.id ? C.blueDim : C.panelAlt,
              border: `1px solid ${selected === c.id ? C.blue : C.border}`,
            }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[13px] font-mono" style={{ color: C.text }}>{c.id}</span>
                {c.selected && <Badge color={C.green}>selected</Badge>}
              </div>
              <MiniGrid seed={c.seed} cell={9} />
              <div className="grid grid-cols-2 gap-2 mt-3 text-[11px]" style={{ color: C.textDim }}>
                <div>Difficulty <span className="font-mono float-right" style={{ color: C.text }}>{c.difficulty.toFixed(2)}</span></div>
                <div>Alignment <span className="font-mono float-right" style={{ color: C.text }}>{c.alignment.toFixed(2)}</span></div>
                <div>Novelty <span className="font-mono float-right" style={{ color: C.text }}>{c.novelty.toFixed(2)}</span></div>
                <div>Validity <span className="font-mono float-right" style={{ color: C.text }}>{c.validity.toFixed(2)}</span></div>
              </div>
            </button>
          ))}
        </div>
      </Panel>

      <div className="flex flex-col gap-4">
        <Panel title={`Detail — ${sel.id}`}>
          <MiniGrid seed={sel.seed} cell={14} />
          <div className="flex flex-col gap-2 mt-3">
            <div>
              <div className="flex justify-between text-[11px] mb-1" style={{ color: C.textDim }}><span>Weakness alignment</span><span className="font-mono">{Math.round(sel.alignment * 100)}%</span></div>
              <Bar1 value={sel.alignment} color={C.orange} />
            </div>
            <div>
              <div className="flex justify-between text-[11px] mb-1" style={{ color: C.textDim }}><span>Novelty</span><span className="font-mono">{Math.round(sel.novelty * 100)}%</span></div>
              <Bar1 value={sel.novelty} color={C.purple} />
            </div>
            <div>
              <div className="flex justify-between text-[11px] mb-1" style={{ color: C.textDim }}><span>Validity</span><span className="font-mono">{Math.round(sel.validity * 100)}%</span></div>
              <Bar1 value={sel.validity} color={C.green} />
            </div>
          </div>
          <button className="w-full mt-3 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-[12px] font-medium" style={{ background: C.blue, color: "#04152e" }}>
            <CheckCircle2 size={13} /> Select this environment
          </button>
        </Panel>
        <Panel title="Expected outcome">
          <KV label="Expected learning progress" value="+8.4%" valueColor={C.green} />
        </Panel>
      </div>
    </div>
  );
}
