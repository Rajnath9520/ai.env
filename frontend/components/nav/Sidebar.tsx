"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutGrid,
  FlaskConical,
  Radio,
  Boxes,
  AlertTriangle,
  GitBranch,
  BarChart3,
  Scale,
  Sparkles,
} from "lucide-react";

function navItems(experimentId?: number) {
  return [
    { href: "/dashboard", label: "Lab Overview", icon: LayoutGrid },
    { href: "/experiments/new", label: "New Experiment", icon: FlaskConical },
    { href: experimentId ? `/live-training/${experimentId}` : "/dashboard", label: "Live", icon: Radio },
    { href: experimentId ? `/environments/${experimentId}` : "/dashboard", label: "Environments", icon: Boxes },
    {
      href: experimentId ? `/failure-analysis/${experimentId}` : "/dashboard",
      label: "Failure Analysis",
      icon: AlertTriangle,
    },
    { href: experimentId ? `/curriculum/${experimentId}` : "/dashboard", label: "Curriculum", icon: GitBranch },
    {
      href: experimentId ? `/generalization/${experimentId}` : "/dashboard",
      label: "Evaluation",
      icon: BarChart3,
    },
    { href: "/comparison", label: "Comparison", icon: Scale },
  ];
}

export function Sidebar({ experimentId }: { experimentId?: number }) {
  const pathname = usePathname();
  const items = navItems(experimentId);

  return (
    <aside className="w-56 shrink-0 bg-[#0b1120] border-r border-app-border flex flex-col h-screen sticky top-0">
      <div className="px-5 py-5 border-b border-app-border">
        <Link href="/dashboard" className="flex items-center gap-2">
          {/* <div className="w-7 h-7 rounded bg-app-blue/15 border border-app-blue/40 flex items-center justify-center">
            <Sparkles size={14} className="text-app-blue" />
          </div> */}
          <div>
            <div className="text-xs font-bold tracking-widest text-app-hi">AI LAB</div>
            <div className="text-[10px] text-app-low tracking-wide">ENVIRONMENT LAB</div>
          </div>
        </Link>
      </div>
      <nav className="flex-1 py-3">
        {items.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || (href !== "/dashboard" && pathname.startsWith(href));
          return (
            <Link
              key={label}
              href={href}
              className={`w-full flex items-center gap-3 px-5 py-2.5 text-[13px] border-l-2 transition-colors ${
                active
                  ? "bg-[#132038] border-app-blue text-app-hi"
                  : "border-transparent text-app-mid hover:bg-app-panel hover:text-app-hi"
              }`}
            >
              <Icon size={15} />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
