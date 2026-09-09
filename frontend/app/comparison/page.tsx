"use client";

import { useEffect, useState } from "react";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { StatusBadge, AgentModeBadge } from "@/components/panels/Primitives";
import { api } from "@/lib/api-client";
import type { ComparisonEntry, ExperimentRead } from "@/types/api";

export default function ComparisonPage() {
  const [experiments, setExperiments] = useState<ExperimentRead[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [entries, setEntries] = useState<ComparisonEntry[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.listExperiments().then((list) => {
      setExperiments(list);
      const completed = list.filter((e) => e.status === "completed").map((e) => e.id);
      setSelected(new Set(completed.slice(0, 4)));
    });
  }, []);

  async function runComparison() {
    if (selected.size === 0) return;
    setLoading(true);
    try {
      const ids = [...selected];
      const result = await api.getComparison(ids[0], ids);
      setEntries(result);
    } finally {
      setLoading(false);
    }
  }

  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <Shell title="COMPARISON" subtitle="Compare curriculum strategies across experiments">
      <Panel title="SELECT EXPERIMENTS">
        {experiments.length === 0 ? (
          <div className="text-app-mid text-xs">No experiments yet.</div>
        ) : (
          <div className="space-y-1.5 max-h-64 overflow-auto">
            {experiments.map((e) => (
              <label key={e.id} className="flex items-center gap-3 text-[12px] px-1 py-1 rounded hover:bg-app-panel2">
                <input type="checkbox" checked={selected.has(e.id)} onChange={() => toggle(e.id)} className="accent-app-blue" />
                <span className="w-8 font-mono text-app-mid">{e.id}</span>
                <span className="flex-1 text-app-hi">{e.name}</span>
                <AgentModeBadge agentMode={e.agent_mode} />
                <span className="text-app-mid">{e.curriculum_type}</span>
                <StatusBadge status={e.status} />
              </label>
            ))}
          </div>
        )}
        <button
          onClick={runComparison}
          disabled={selected.size === 0 || loading}
          className="mt-3 px-3 py-1.5 rounded bg-app-blue text-white text-[12px] font-semibold hover:bg-[#2f78ea] disabled:opacity-50"
        >
          {loading ? "Loading..." : "Compare Selected"}
        </button>
      </Panel>

      {entries && (
        <Panel title="RESULTS">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left text-app-low">
                <th className="pb-2 font-medium">Experiment</th>
                <th className="pb-2 font-medium">Agent</th>
                <th className="pb-2 font-medium">Curriculum</th>
                <th className="pb-2 font-medium">Status</th>
                <th className="pb-2 font-medium">Final Success Rate</th>
                <th className="pb-2 font-medium">Validation</th>
                <th className="pb-2 font-medium">Test</th>
                <th className="pb-2 font-medium">OOD</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.experiment_id} className="border-t border-[#1a2033]">
                  <td className="py-1.5 text-app-hi">{entry.name}</td>
                  <td className="py-1.5">
                    <AgentModeBadge agentMode={entry.agent_mode} />
                  </td>
                  <td className="py-1.5 text-app-mid">{entry.curriculum_type}</td>
                  <td className="py-1.5">
                    <StatusBadge status={entry.status} />
                  </td>
                  <td className="py-1.5 font-mono text-app-hi">
                    {entry.final_success_rate != null ? entry.final_success_rate.toFixed(2) : "--"}
                  </td>
                  {(["validation", "test", "ood"] as const).map((split) => (
                    <td key={split} className="py-1.5 font-mono text-app-hi">
                      {entry.generalization[split] ? entry.generalization[split].success_rate.toFixed(2) : "--"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
    </Shell>
  );
}
