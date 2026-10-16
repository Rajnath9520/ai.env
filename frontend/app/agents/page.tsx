"use client";

import { useEffect, useState, type FormEvent } from "react";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { AgentStatusBadge } from "@/components/panels/Primitives";
import { api, extractErrorMessage } from "@/lib/api-client";
import type { AgentRead } from "@/types/api";

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentRead[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [token, setToken] = useState("");
  const [registering, setRegistering] = useState(false);
  const [registerError, setRegisterError] = useState<string | null>(null);

  const [pingingId, setPingingId] = useState<number | null>(null);
  const [tokenEditId, setTokenEditId] = useState<number | null>(null);
  const [tokenDraft, setTokenDraft] = useState("");
  const [tokenError, setTokenError] = useState<string | null>(null);

  async function refresh() {
    setAgents(await api.listLocalAgents());
  }

  useEffect(() => {
    refresh().catch((e) => setLoadError(extractErrorMessage(e, "Failed to load agents")));
  }, []);

  async function handleRegister(e: FormEvent) {
    e.preventDefault();
    setRegistering(true);
    setRegisterError(null);
    try {
      await api.registerLocalAgent({
        name: name.trim(),
        base_url: baseUrl.trim(),
        token: token.trim() || null,
      });
      setName("");
      setBaseUrl("");
      setToken("");
      await refresh();
    } catch (err) {
      setRegisterError(extractErrorMessage(err, "Failed to register agent"));
    } finally {
      setRegistering(false);
    }
  }

  async function handlePing(id: number) {
    setPingingId(id);
    try {
      await api.pingLocalAgent(id);
      await refresh();
    } catch {
      // the row's own status will already say "unreachable" if this failed
    } finally {
      setPingingId(null);
    }
  }

  async function handleSaveToken(id: number) {
    setTokenError(null);
    try {
      await api.updateLocalAgentToken(id, tokenDraft.trim() || null);
      setTokenEditId(null);
      setTokenDraft("");
      await refresh();
    } catch (err) {
      setTokenError(extractErrorMessage(err, "Failed to update token"));
    }
  }

  return (
    <Shell
      title="AGENTS"
      subtitle="Section 5's Local Agent -- a separate process, typically on your own machine, that a project's scan/validate/run can be proxied to instead of this backend executing them directly"
    >
      {loadError && <div className="text-app-red text-xs">{loadError}</div>}

      <Panel title="REGISTER AN AGENT">
        <div className="text-[11px] text-app-low mb-3">
          Start one with{" "}
          <code className="text-app-hi">
            AI_ENV_AGENT_TOKEN=&lt;secret&gt; uvicorn agent_app.main:app --port 8765
          </code>{" "}
          on the machine that should run this, then enter the same secret below. An agent with no token
          configured refuses to run anything. The token is a shared secret sent over plain HTTP, so use a
          trusted network or put TLS in front of the agent -- see docs/local-agent.md. The token is stored by
          the backend to authenticate to the agent and is never shown again.
        </div>
        <form onSubmit={handleRegister} className="flex items-end gap-3">
          <label className="flex flex-col gap-1 text-xs flex-1">
            <span className="text-app-mid">Name</span>
            <input
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my-laptop"
              className="bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi focus:outline-none focus:border-app-blue"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs flex-[2]">
            <span className="text-app-mid">Base URL</span>
            <input
              required
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://192.168.1.42:8765"
              className="bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi font-mono focus:outline-none focus:border-app-blue"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs flex-1">
            <span className="text-app-mid">Token</span>
            <input
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="agent's AI_ENV_AGENT_TOKEN"
              className="bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi font-mono focus:outline-none focus:border-app-blue"
            />
          </label>
          <button
            type="submit"
            disabled={registering}
            className="px-4 py-1.5 rounded bg-app-blue text-white text-[12px] font-semibold hover:bg-[#2f78ea] disabled:opacity-50"
          >
            {registering ? "Registering..." : "Register"}
          </button>
        </form>
        {registerError && <div className="text-app-red text-[11px] mt-2">{registerError}</div>}
      </Panel>

      <Panel title="REGISTERED AGENTS">
        {!agents ? (
          <div className="text-app-mid text-xs">Loading...</div>
        ) : agents.length === 0 ? (
          <div className="text-app-mid text-xs">
            No agents registered yet -- every project executes directly on this backend until you register
            one and assign it.
          </div>
        ) : (
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left text-app-low">
                <th className="pb-2 font-medium">ID</th>
                <th className="pb-2 font-medium">Name</th>
                <th className="pb-2 font-medium">Base URL</th>
                <th className="pb-2 font-medium">Token</th>
                <th className="pb-2 font-medium">Status</th>
                <th className="pb-2 font-medium">Last seen</th>
                <th className="pb-2 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {agents.map((a) => (
                <tr key={a.id} className="border-t border-[#1a2033]">
                  <td className="py-1.5 font-mono text-app-mid">{a.id}</td>
                  <td className="py-1.5 text-app-hi">{a.name}</td>
                  <td className="py-1.5 text-app-hi font-mono">{a.base_url}</td>
                  <td className="py-1.5 text-app-mid">{a.has_token ? "stored" : "none"}</td>
                  <td className="py-1.5">
                    <AgentStatusBadge status={a.status} />
                    {a.status === "unauthorized" && (
                      <div className="text-[10px] text-app-orange mt-0.5">
                        reachable, but the token is missing or wrong
                      </div>
                    )}
                  </td>
                  <td className="py-1.5 text-app-mid">
                    {a.last_seen_at ? new Date(a.last_seen_at).toLocaleString() : "never"}
                  </td>
                  <td className="py-1.5 text-right space-x-1.5">
                    {tokenEditId === a.id ? (
                      <span className="inline-flex items-center gap-1.5">
                        <input
                          type="password"
                          autoComplete="off"
                          autoFocus
                          value={tokenDraft}
                          onChange={(e) => setTokenDraft(e.target.value)}
                          placeholder="new token"
                          className="w-32 bg-app-panel2 border border-app-border rounded px-1.5 py-0.5 text-app-hi font-mono text-[11px] focus:outline-none focus:border-app-blue"
                        />
                        <button
                          onClick={() => handleSaveToken(a.id)}
                          className="px-2 py-0.5 rounded bg-app-blue text-white text-[10px] font-semibold"
                        >
                          Save
                        </button>
                        <button
                          onClick={() => {
                            setTokenEditId(null);
                            setTokenDraft("");
                            setTokenError(null);
                          }}
                          className="px-2 py-0.5 rounded border border-app-border text-[10px] text-app-mid"
                        >
                          Cancel
                        </button>
                      </span>
                    ) : (
                      <button
                        onClick={() => {
                          setTokenEditId(a.id);
                          setTokenDraft("");
                        }}
                        className="px-2 py-0.5 rounded border border-app-border text-[10px] text-app-mid hover:border-app-blue"
                      >
                        Set token
                      </button>
                    )}
                    <button
                      onClick={() => handlePing(a.id)}
                      disabled={pingingId === a.id}
                      className="px-2 py-0.5 rounded border border-app-border text-[10px] text-app-mid hover:border-app-blue disabled:opacity-50"
                    >
                      {pingingId === a.id ? "Pinging..." : "Ping"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {tokenError && <div className="text-app-red text-[11px] mt-2">{tokenError}</div>}
      </Panel>
    </Shell>
  );
}
