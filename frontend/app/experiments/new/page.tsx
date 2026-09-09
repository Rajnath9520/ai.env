"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { api, ApiError } from "@/lib/api-client";
import {
  DEFAULT_EXPERIMENT_CREATE,
  type AgentMode,
  type CurriculumType,
  type ExperimentCreate,
  type GenerationStrategy,
} from "@/types/api";

const CURRICULUM_OPTIONS: { value: CurriculumType; label: string; description: string }[] = [
  { value: "random", label: "Random", description: "Baseline A -- fully random environment parameters each step." },
  { value: "fixed", label: "Fixed", description: "Baseline B -- predetermined easy/moderate/hard/very-hard stages." },
  {
    value: "generic_adaptive",
    label: "Generic Adaptive",
    description: "Baseline C -- adjusts difficulty from overall success rate only.",
  },
  {
    value: "failure_driven",
    label: "Failure Driven",
    description: "Proposed method -- targets environments at the agent's detected weaknesses.",
  },
];

// Phase 28 (Section 14) -- the three explicit agent modes. Environment
// Only is the default; nothing here silently trains a model.
const AGENT_MODE_OPTIONS: { value: AgentMode; label: string; description: string }[] = [
  {
    value: "environment_only",
    label: "Environment Only",
    description:
      "Generate, validate, analyze, compare, and export environments without training an AI model.",
  },
  {
    value: "external",
    label: "External Agent",
    description: "Connect an external agent and analyze its behavior and failures.",
  },
  {
    value: "demo_ppo",
    label: "Demo Agent -- PPO",
    description: "Run the built-in PPO demonstration agent.",
  },
];

const GENERATION_STRATEGY_OPTIONS: { value: GenerationStrategy; label: string; description: string }[] = [
  {
    value: "diverse",
    label: "Diverse",
    description: "Generate several candidates per step, keep the most different valid one.",
  },
  { value: "progressive", label: "Progressive", description: "Follow a fixed easy-to-hard schedule." },
  { value: "random", label: "Random", description: "Fully random parameters each step." },
];

function NumberField({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-app-mid">{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
        className="bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi font-mono focus:outline-none focus:border-app-blue"
      />
    </label>
  );
}

export default function NewExperimentPage() {
  const router = useRouter();
  const [form, setForm] = useState<ExperimentCreate>(DEFAULT_EXPERIMENT_CREATE);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function set<K extends keyof ExperimentCreate>(key: K, value: ExperimentCreate[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const created = await api.createExperiment(form);
      router.push(`/experiments/${created.id}`);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(
          typeof err.body === "object" && err.body && "detail" in (err.body as Record<string, unknown>)
            ? String((err.body as Record<string, unknown>).detail)
            : `Request failed (${err.status})`,
        );
      } else {
        setError("Failed to create experiment");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Shell title="CREATE EXPERIMENT" subtitle="Configure a new curriculum run">
      <form onSubmit={handleSubmit} className="space-y-4 max-w-3xl">
        <Panel title="BASICS">
          <div className="grid grid-cols-2 gap-4">
            <label className="flex flex-col gap-1 text-xs">
              <span className="text-app-mid">Name</span>
              <input
                required
                value={form.name}
                onChange={(e) => set("name", e.target.value)}
                placeholder="e.g. failure-driven-run-1"
                className="bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi focus:outline-none focus:border-app-blue"
              />
            </label>
            <NumberField label="Seed" value={form.seed} onChange={(v) => set("seed", v)} />
          </div>
        </Panel>

        <Panel title="AGENT">
          <div className="grid grid-cols-3 gap-3">
            {AGENT_MODE_OPTIONS.map((opt) => (
              <button
                type="button"
                key={opt.value}
                onClick={() => set("agent_mode", opt.value)}
                className={`text-left p-3 rounded border transition-colors ${
                  form.agent_mode === opt.value
                    ? "border-app-blue bg-[#132038]"
                    : "border-app-border bg-app-panel2 hover:border-app-mid"
                }`}
              >
                <div className="text-[13px] font-semibold text-app-hi">{opt.label}</div>
                <div className="text-[11px] text-app-mid mt-1">{opt.description}</div>
              </button>
            ))}
          </div>
        </Panel>

        {form.agent_mode === "demo_ppo" && (
          <Panel title="CURRICULUM STRATEGY">
            <div className="grid grid-cols-2 gap-3">
              {CURRICULUM_OPTIONS.map((opt) => (
                <button
                  type="button"
                  key={opt.value}
                  onClick={() => set("curriculum_type", opt.value)}
                  className={`text-left p-3 rounded border transition-colors ${
                    form.curriculum_type === opt.value
                      ? "border-app-blue bg-[#132038]"
                      : "border-app-border bg-app-panel2 hover:border-app-mid"
                  }`}
                >
                  <div className="text-[13px] font-semibold text-app-hi">{opt.label}</div>
                  <div className="text-[11px] text-app-mid mt-1">{opt.description}</div>
                </button>
              ))}
            </div>
          </Panel>
        )}

        {form.agent_mode === "environment_only" && (
          <Panel title="GENERATION STRATEGY">
            <div className="grid grid-cols-3 gap-3">
              {GENERATION_STRATEGY_OPTIONS.map((opt) => (
                <button
                  type="button"
                  key={opt.value}
                  onClick={() => set("generation_strategy", opt.value)}
                  className={`text-left p-3 rounded border transition-colors ${
                    form.generation_strategy === opt.value
                      ? "border-app-blue bg-[#132038]"
                      : "border-app-border bg-app-panel2 hover:border-app-mid"
                  }`}
                >
                  <div className="text-[13px] font-semibold text-app-hi">{opt.label}</div>
                  <div className="text-[11px] text-app-mid mt-1">{opt.description}</div>
                </button>
              ))}
            </div>
          </Panel>
        )}

        <Panel title="ENVIRONMENT">
          <div className="grid grid-cols-3 gap-4">
            <NumberField label="Grid Size" value={form.grid_size} min={4} max={24} onChange={(v) => set("grid_size", v)} />
            <NumberField
              label="Obstacle Density"
              value={form.obstacle_density}
              step={0.05}
              min={0}
              max={0.9}
              onChange={(v) => set("obstacle_density", v)}
            />
            <NumberField
              label="Dead-End Probability"
              value={form.dead_end_probability}
              step={0.05}
              min={0}
              max={1}
              onChange={(v) => set("dead_end_probability", v)}
            />
            <NumberField
              label="Path Complexity"
              value={form.path_complexity}
              step={0.05}
              min={0}
              max={1}
              onChange={(v) => set("path_complexity", v)}
            />
            <NumberField
              label="Max Steps / Episode"
              value={form.max_steps_per_episode}
              onChange={(v) => set("max_steps_per_episode", v)}
            />
          </div>
        </Panel>

        {form.agent_mode === "environment_only" && (
          <Panel title="GENERATION">
            <div className="grid grid-cols-3 gap-4">
              <NumberField
                label="Environments to Generate"
                value={form.n_curriculum_steps}
                onChange={(v) => set("n_curriculum_steps", v)}
              />
              {form.generation_strategy === "diverse" && (
                <NumberField
                  label="Candidates / Step"
                  value={form.n_candidates}
                  onChange={(v) => set("n_candidates", v)}
                />
              )}
            </div>
          </Panel>
        )}

        {form.agent_mode === "external" && (
          <Panel title="ADAPTIVE GENERATION">
            <div className="grid grid-cols-2 gap-4">
              <NumberField label="Candidates / Step" value={form.n_candidates} onChange={(v) => set("n_candidates", v)} />
              <NumberField
                label="Trajectories / Analysis"
                value={form.trajectories_per_analysis}
                onChange={(v) => set("trajectories_per_analysis", v)}
              />
            </div>
            <p className="text-[11px] text-app-mid mt-3">
              An external agent connects to this experiment&apos;s reset/step interface after it starts. No model is
              created by ai.env in this mode -- the connected agent owns its own model.
            </p>
          </Panel>
        )}

        {form.agent_mode === "demo_ppo" && (
          <Panel title="TRAINING">
            <div className="grid grid-cols-3 gap-4">
              <NumberField
                label="Curriculum Steps"
                value={form.n_curriculum_steps}
                onChange={(v) => set("n_curriculum_steps", v)}
              />
              <NumberField
                label="Timesteps / Step"
                value={form.timesteps_per_step}
                step={1000}
                onChange={(v) => set("timesteps_per_step", v)}
              />
              <NumberField label="Eval Episodes" value={form.eval_episodes} onChange={(v) => set("eval_episodes", v)} />
              {form.curriculum_type === "failure_driven" && (
                <>
                  <NumberField label="Candidates / Step" value={form.n_candidates} onChange={(v) => set("n_candidates", v)} />
                  <NumberField
                    label="Trajectories / Analysis"
                    value={form.trajectories_per_analysis}
                    onChange={(v) => set("trajectories_per_analysis", v)}
                  />
                </>
              )}
            </div>
          </Panel>
        )}

        {form.agent_mode === "demo_ppo" && (
          <Panel title="GENERALIZATION EVALUATION">
            <div className="flex items-center gap-3 mb-3">
              <input
                type="checkbox"
                id="run_generalization"
                checked={form.run_generalization}
                onChange={(e) => set("run_generalization", e.target.checked)}
                className="accent-app-blue"
              />
              <label htmlFor="run_generalization" className="text-xs text-app-hi">
                Run generalization evaluation (validation/test/OOD) after training
              </label>
            </div>
            {form.run_generalization && (
              <div className="grid grid-cols-2 gap-4">
                <NumberField
                  label="Environments / Split"
                  value={form.generalization_envs_per_split}
                  onChange={(v) => set("generalization_envs_per_split", v)}
                />
                <NumberField
                  label="Episodes / Environment"
                  value={form.generalization_episodes_per_config}
                  onChange={(v) => set("generalization_episodes_per_config", v)}
                />
              </div>
            )}
          </Panel>
        )}

        {error && <div className="text-app-red text-xs">{error}</div>}

        <button
          type="submit"
          disabled={submitting}
          className="px-4 py-2 rounded bg-app-blue text-white text-[13px] font-semibold hover:bg-[#2f78ea] disabled:opacity-50"
        >
          {submitting ? "Creating..." : "Create Experiment"}
        </button>
      </form>
    </Shell>
  );
}
