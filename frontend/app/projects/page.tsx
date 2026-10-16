"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { ProjectStatusBadge } from "@/components/panels/Primitives";
import { api, extractErrorMessage } from "@/lib/api-client";
import type { ProjectRead } from "@/types/api";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectRead[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listProjects()
      .then((list) => !cancelled && setProjects(list))
      .catch((e) => !cancelled && setError(extractErrorMessage(e, "Failed to load projects")));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Shell
      title="PROJECTS"
      subtitle="Your own AI projects, registered and orchestrated through the ai-env.yaml contract"
      actions={
        <Link
          href="/projects/new"
          className="px-3 py-1.5 rounded bg-app-blue text-white text-[12px] font-semibold hover:bg-[#2f78ea]"
        >
          Register AI Project
        </Link>
      }
    >
      {error && <div className="text-app-red text-xs">{error}</div>}

      <Panel title="REGISTERED PROJECTS">
        {!projects ? (
          <div className="text-app-mid text-xs">Loading...</div>
        ) : projects.length === 0 ? (
          <div className="text-app-mid text-xs">
            Register your first AI project to begin adaptive training.{" "}
            <Link href="/projects/new" className="text-app-blue">
              Register a project
            </Link>
            .
          </div>
        ) : (
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left text-app-low">
                <th className="pb-2 font-medium">ID</th>
                <th className="pb-2 font-medium">Name</th>
                <th className="pb-2 font-medium">Framework</th>
                <th className="pb-2 font-medium">Task</th>
                <th className="pb-2 font-medium">Source</th>
                <th className="pb-2 font-medium">Status</th>
                <th className="pb-2 font-medium">Registered</th>
              </tr>
            </thead>
            <tbody>
              {projects
                .slice()
                .sort((a, b) => b.id - a.id)
                .map((p) => (
                  <tr key={p.id} className="border-t border-[#1a2033] hover:bg-app-panel2">
                    <td className="py-1.5 font-mono text-app-mid">{p.id}</td>
                    <td className="py-1.5">
                      <Link href={`/projects/${p.id}`} className="text-app-blue hover:underline">
                        {p.name}
                      </Link>
                    </td>
                    <td className="py-1.5 text-app-hi">{p.framework ?? "--"}</td>
                    <td className="py-1.5 text-app-hi">{p.task ?? "--"}</td>
                    <td className="py-1.5 text-app-mid uppercase text-[10px] tracking-wide">{p.source}</td>
                    <td className="py-1.5">
                      <ProjectStatusBadge status={p.status} />
                    </td>
                    <td className="py-1.5 text-app-mid">{new Date(p.created_at).toLocaleString()}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </Panel>
    </Shell>
  );
}
