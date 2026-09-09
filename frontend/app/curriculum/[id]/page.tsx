"use client";

import { useEffect, useState } from "react";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { api } from "@/lib/api-client";
import type { EnvironmentRead, ExperimentRead, LearningProgressRead } from "@/types/api";

export default function CurriculumPage({ params }: { params: { id: string } }) {
  const experimentId = Number(params.id);
  const [environments, setEnvironments] = useState<EnvironmentRead[]>([]);
  const [learningProgress, setLearningProgress] = useState<LearningProgressRead[]>([]);
  const [experiment, setExperiment] = useState<ExperimentRead | null>(null);

  useEffect(() => {
    api.getEnvironments(experimentId).then(setEnvironments);
    api.getLearningProgress(experimentId).then(setLearningProgress);
    api.getExperiment(experimentId).then(setExperiment);
  }, [experimentId]);

  const lpByEnvId = new Map(learningProgress.map((lp) => [lp.environment_id, lp]));
  const hasAgent = experiment?.agent_mode === "demo_ppo";

  return (
    <Shell experimentId={experimentId} title="CURRICULUM TIMELINE" subtitle="Environment sequence, target weakness, and learning progress per step">
      {experiment && !hasAgent && (
        <div className="text-app-mid text-xs">
          {experiment.agent_mode === "environment_only"
            ? "This experiment's agent_mode is Environment Only -- no agent ran, so Success Rate and Learning Progress below are always \"--\"."
            : "This experiment's agent is external -- ai.env doesn't compute Success Rate or Learning Progress for it here."}
        </div>
      )}
      <Panel title="TIMELINE">
        {environments.length === 0 ? (
          <div className="text-app-mid text-xs">No curriculum steps yet.</div>
        ) : (
          <>
            <div className="flex items-center gap-1 overflow-x-auto pb-3">
              {environments.map((env, i) => (
                <div key={env.id} className="flex items-center gap-1 shrink-0">
                  <div className="flex flex-col items-center w-20">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ background: env.difficulty > 0.6 ? "#f97a2c" : "#3f8cff" }}
                    />
                    <span className="text-[9px] text-app-mid mt-1 truncate w-full text-center">
                      {env.environment_key}
                    </span>
                    <span className="text-[9px] font-mono text-app-hi">{env.difficulty.toFixed(2)}</span>
                  </div>
                  {i < environments.length - 1 && <div className="h-px w-4 bg-app-border shrink-0" />}
                </div>
              ))}
            </div>

            <table className="w-full text-[12px] mt-2">
              <thead>
                <tr className="text-left text-app-low">
                  <th className="pb-2 font-medium">Environment</th>
                  <th className="pb-2 font-medium">Target Weakness</th>
                  <th className="pb-2 font-medium">Difficulty</th>
                  <th className="pb-2 font-medium">Success Rate</th>
                  <th className="pb-2 font-medium">Learning Progress</th>
                </tr>
              </thead>
              <tbody>
                {environments.map((env) => {
                  const lp = lpByEnvId.get(env.id);
                  return (
                    <tr key={env.id} className="border-t border-[#1a2033]">
                      <td className="py-1.5 font-mono text-app-mid">{env.environment_key}</td>
                      <td className="py-1.5 text-app-orange">{env.target_weakness ?? "--"}</td>
                      <td className="py-1.5 font-mono text-app-hi">{env.difficulty.toFixed(2)}</td>
                      <td className="py-1.5 font-mono text-app-hi">
                        {lp ? lp.performance_after.toFixed(2) : "--"}
                      </td>
                      <td
                        className="py-1.5 font-mono"
                        style={{ color: lp ? (lp.learning_progress >= 0 ? "#22c55e" : "#ef4444") : "#8b96ab" }}
                      >
                        {lp ? `${lp.learning_progress >= 0 ? "+" : ""}${lp.learning_progress.toFixed(3)}` : "--"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}
      </Panel>
    </Shell>
  );
}
