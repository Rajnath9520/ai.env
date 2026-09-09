import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend } from "recharts";
import { C } from "../theme.js";
import { comparisonData, generalization } from "../lib/mockData.js";

const chartText = { fontSize: 11, fill: C.textFaint };

export function SuccessProgressChart({ data, height = 220 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={C.border} vertical={false} />
        <XAxis dataKey="ep" tick={chartText} axisLine={{ stroke: C.border }} tickLine={false} />
        <YAxis yAxisId="l" tick={chartText} axisLine={false} tickLine={false} domain={[0, 100]} />
        <YAxis yAxisId="r" orientation="right" tick={chartText} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={{ background: C.panelAlt, border: `1px solid ${C.border}`, fontSize: 12, color: C.text }} />
        <Line yAxisId="l" type="monotone" dataKey="success" name="Success rate" stroke={C.blue} strokeWidth={2} dot={false} />
        <Line yAxisId="r" type="monotone" dataKey="progress" name="Learning progress" stroke={C.green} strokeWidth={1.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ComparisonChart({ height = 260 }) {
  const lineColors = { Random: C.textFaint, Fixed: C.yellow, "Generic adaptive": C.purple, "Failure driven": C.orange };
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={comparisonData} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={C.border} vertical={false} />
        <XAxis dataKey="ep" tick={chartText} axisLine={{ stroke: C.border }} tickLine={false} />
        <YAxis tick={chartText} axisLine={false} tickLine={false} domain={[0, 100]} />
        <Tooltip contentStyle={{ background: C.panelAlt, border: `1px solid ${C.border}`, fontSize: 12, color: C.text }} />
        <Legend wrapperStyle={{ fontSize: 12, color: C.textDim }} />
        {Object.keys(lineColors).map((k) => (
          <Line key={k} type="monotone" dataKey={k} stroke={lineColors[k]} strokeWidth={k === "Failure driven" ? 2.5 : 1.5} dot={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function GeneralizationChart({ height = 240 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={generalization} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={C.border} vertical={false} />
        <XAxis dataKey="split" tick={chartText} axisLine={{ stroke: C.border }} tickLine={false} />
        <YAxis tick={chartText} axisLine={false} tickLine={false} domain={[0, 100]} />
        <Tooltip contentStyle={{ background: C.panelAlt, border: `1px solid ${C.border}`, fontSize: 12, color: C.text }} />
        <Bar dataKey="success" name="Success rate" fill={C.blue} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
