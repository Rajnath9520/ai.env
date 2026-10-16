"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { api, extractErrorMessage } from "@/lib/api-client";

export default function RegisterProjectPage() {
  const router = useRouter();
  const [rootPath, setRootPath] = useState("");
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const created = await api.registerProject({
        root_path: rootPath.trim(),
        name: name.trim() || null,
      });
      router.push(`/projects/${created.id}`);
    } catch (err) {
      setError(extractErrorMessage(err, "Failed to register project"));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Shell title="REGISTER AI PROJECT" subtitle="Connect an existing AI project built outside ai.env">
      <form onSubmit={handleSubmit} className="space-y-4 max-w-xl">
        <Panel title="PROJECT LOCATION">
          <div className="space-y-4">
            <label className="flex flex-col gap-1 text-xs">
              <span className="text-app-mid">Project folder path</span>
              <input
                required
                value={rootPath}
                onChange={(e) => setRootPath(e.target.value)}
                placeholder="/home/you/projects/face-recognition"
                className="bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi font-mono focus:outline-none focus:border-app-blue"
              />
              <span className="text-[11px] text-app-low">
                A path the ai.env backend process can read directly. There is no Local Agent yet
                (docs/projects-and-adapters.md), so this only works when ai.env and your project are on the
                same machine -- not a path on your own laptop if ai.env is running elsewhere.
              </span>
            </label>

            <label className="flex flex-col gap-1 text-xs">
              <span className="text-app-mid">Name (optional)</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Defaults to the folder name"
                className="bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi focus:outline-none focus:border-app-blue"
              />
            </label>
          </div>
        </Panel>

        {error && <div className="text-app-red text-xs">{error}</div>}

        <button
          type="submit"
          disabled={submitting}
          className="px-4 py-2 rounded bg-app-blue text-white text-[12px] font-semibold hover:bg-[#2f78ea] disabled:opacity-50"
        >
          {submitting ? "Registering..." : "Register Project"}
        </button>
      </form>
    </Shell>
  );
}
