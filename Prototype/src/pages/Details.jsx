import React, { useState } from "react";
import { Download } from "lucide-react";
import { C } from "../theme.js";
import { Panel, Bar1, KV, StatusDot } from "../components/ui.jsx";
import { SuccessProgressChart } from "../components/charts.jsx";
import { curriculumEnvs, failureLog, weaknesses, learningProgressData } from "../lib/mockData.js";

const tabs = ["Configuration", "Training history", "Environment history", "Failure history", "Weakness history", "Checkpoints"];

export default function DetailsPage() {
  const [tab, setTab] = useState(tabs[0]);
  return (
    <div className="p-5 flex flex-col gap-4">
      <Panel title="EXP-024 — failure driven curriculum">
        <div className="grid grid-cols-4 gap-4">
          <KV label="Algorithm" value="PPO" />
          <KV label="Environment" value="GridWorld 12×12" />
          <KV label="Seed" value="48291" />
          <KV label="Status" value={<StatusDot status="training" />} />
        </div>
      </Panel>

      <div className="flex items-center gap-1 overflow-x-auto">
        {tabs.map((t) => (
          <button key={t} onClick={() => setTab(t)} className="px-3 py-1.5 rounded-md text-[12px] whitespace-nowrap" style={{
            background: tab === t ? C.blueDim : "transparent",
            color: tab === t ? "#bcd6ff" : C.textDim,
            border: `1px solid ${tab === t ? C.blue : C.border}`,
          }}>{t}</button>
        ))}
      </div>

      {tab === "Configuration" && (
        <Panel title="Configuration snapshot">
          <pre className="text-[12px] font-mono whitespace-pre-wrap" style={{ color: C.textDim }}>{`algorithm: PPO
curriculum:
  type: failure_driven
environment:
  type: gridworld
  size: 12
training:
  episodes: 10000
evaluation:
  frequency: 500
seed: 48291`}</pre>
        </Panel>
      )}
      {tab === "Training history" && <Panel title="Success rate over training"><SuccessProgressChart data={learningProgressData} /></Panel>}
      {tab === "Environment history" && (
        <Panel pad={false} title="Environments used">
          <table className="w-full text-[13px]">
            <tbody>
              {curriculumEnvs.map((e) => (
                <tr key={e.id} style={{ borderBottom: `1px solid ${C.border}` }}>
                  <td className="px-4 py-2 font-mono" style={{ color: C.text }}>{e.id}</td>
                  <td className="px-4 py-2" style={{ color: C.textDim }}>{e.weakness}</td>
                  <td className="px-4 py-2 font-mono" style={{ color: C.textDim }}>{e.difficulty.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
      {tab === "Failure history" && (
        <Panel pad={false} title="Failure log">
          <table className="w-full text-[13px]">
            <tbody>
              {failureLog.map((f) => (
                <tr key={f.ep} style={{ borderBottom: `1px solid ${C.border}` }}>
                  <td className="px-4 py-2 font-mono" style={{ color: C.textDim }}>{f.ep}</td>
                  <td className="px-4 py-2" style={{ color: C.text }}>{f.type}</td>
                  <td className="px-4 py-2 font-mono" style={{ color: C.red }}>{f.severity.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
      {tab === "Weakness history" && (
        <Panel title="Weakness vector over time">
          <div className="flex flex-col gap-2">
            {weaknesses.map((w) => (
              <div key={w.key}>
                <div className="flex justify-between text-[11px] mb-1" style={{ color: C.textDim }}><span>{w.label}</span><span className="font-mono">{w.value.toFixed(2)}</span></div>
                <Bar1 value={w.value} color={w.color} />
              </div>
            ))}
          </div>
        </Panel>
      )}
      {tab === "Checkpoints" && (
        <Panel pad={false} title="Model checkpoints">
          <table className="w-full text-[13px]">
            <tbody>
              {[1000, 1250, 1500, 1750].map((ep) => (
                <tr key={ep} style={{ borderBottom: `1px solid ${C.border}` }}>
                  <td className="px-4 py-2 font-mono" style={{ color: C.text }}>checkpoint_ep{ep}.pt</td>
                  <td className="px-4 py-2 text-right"><button className="text-[11px] flex items-center gap-1 ml-auto" style={{ color: C.blue }}><Download size={12} />Download</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
    </div>
  );
}
