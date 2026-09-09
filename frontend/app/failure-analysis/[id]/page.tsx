"use client";

import { useEffect, useState } from "react";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { Bar } from "@/components/panels/Primitives";
import { api } from "@/lib/api-client";
import type { FailureRead } from "@/types/api";

function severityColor(score: number): string {
  if (score >= 0.7) return "#ef4444";
  if (score >= 0.4) return "#f97a2c";
  return "#22c55e";
}

export default function FailureAnalysisPage({ params }: { params: { id: string } }) {
  const experimentId = Number(params.id);
  const [failures, setFailures] = useState<FailureRead[] | null>(null);

  useEffect(() => {
    api.getFailures(experimentId).then(setFailures);
  }, [experimentId]);

  const byType = new Map<string, { total: number; count: number }>();
  (failures ?? []).forEach((f) => {
    const entry = byType.get(f.failure_type) ?? { total: 0, count: 0 };
    entry.total += f.score;
    entry.count += 1;
    byType.set(f.failure_type, entry);
  });
  const averaged = [...byType.entries()]
    .map(([type, { total, count }]) => ({ type, avg: total / count }))
    .sort((a, b) => b.avg - a.avg);

  return (
    <Shell experimentId={experimentId} title="FAILURE ANALYSIS" subtitle="Detected behavioral failure modes across all trained environments">
      <div className="grid grid-cols-3 gap-4">
        <Panel title="AVERAGE FAILURE SCORE BY TYPE" className="col-span-1">
          {averaged.length === 0 ? (
            <div className="text-app-mid text-xs">No failure data yet.</div>
          ) : (
            <div className="space-y-2.5">
              {averaged.map((row) => (
                <Bar key={row.type} label={row.type.replace(/_/g, " ")} value={row.avg} color={severityColor(row.avg)} />
              ))}
            </div>
          )}
        </Panel>

        <Panel title="DETECTED FAILURES" className="col-span-2">
          {!failures ? (
            <div className="text-app-mid text-xs">Loading...</div>
          ) : failures.length === 0 ? (
            <div className="text-app-mid text-xs">No failures recorded yet -- this experiment may not have run, or is not a failure_driven curriculum.</div>
          ) : (
            <div className="max-h-[520px] overflow-auto">
              <table className="w-full text-[12px]">
                <thead className="sticky top-0 bg-app-panel">
                  <tr className="text-left text-app-low">
                    <th className="pb-2 font-medium">Type</th>
                    <th className="pb-2 font-medium">Severity</th>
                    <th className="pb-2 font-medium">Confidence</th>
                    <th className="pb-2 font-medium">Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {failures
                    .slice()
                    .sort((a, b) => b.score - a.score)
                    .map((f) => (
                      <tr key={f.id} className="border-t border-[#1a2033] align-top">
                        <td className="py-1.5 text-app-hi whitespace-nowrap">{f.failure_type.replace(/_/g, " ")}</td>
                        <td className="py-1.5 font-mono" style={{ color: severityColor(f.score) }}>
                          {f.score.toFixed(2)}
                        </td>
                        <td className="py-1.5 font-mono text-app-mid">{f.confidence.toFixed(2)}</td>
                        <td className="py-1.5 text-app-mid">
                          {Object.entries(f.features)
                            .slice(0, 3)
                            .map(([k, v]) => `${k}: ${v}`)
                            .join(" · ")}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </Shell>
  );
}
