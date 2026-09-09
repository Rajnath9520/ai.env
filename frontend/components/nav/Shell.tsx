import type { ReactNode } from "react";
import { Sidebar } from "@/components/nav/Sidebar";

export function Shell({
  experimentId,
  title,
  subtitle,
  actions,
  children,
}: {
  experimentId?: number;
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="min-h-screen w-full bg-app-bg flex">
      <Sidebar experimentId={experimentId} />
      <div className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center justify-between px-6 py-4 border-b border-app-border">
          <div>
            <h1 className="text-[13px] font-semibold tracking-widest text-app-hi">{title}</h1>
            {subtitle && <div className="mt-1 text-xs text-app-mid">{subtitle}</div>}
          </div>
          {actions}
        </header>
        <main className="flex-1 overflow-auto p-4 space-y-4">{children}</main>
      </div>
    </div>
  );
}
