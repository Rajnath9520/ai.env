import type { ReactNode } from "react";

export function Panel({
  title,
  right,
  children,
  className = "",
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`bg-app-panel border border-app-border rounded-md flex flex-col ${className}`}>
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-app-border">
        <h3 className="text-[11px] font-semibold tracking-widest text-app-mid">{title}</h3>
        {right}
      </div>
      <div className="p-4 flex-1 min-h-0">{children}</div>
    </div>
  );
}
