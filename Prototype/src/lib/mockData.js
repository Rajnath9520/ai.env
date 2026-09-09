import { C } from "../theme.js";

/* ---------------- deterministic RNG / gridworld / curves ---------------- */
export function makeRng(seed) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}

export function makeGrid(seed, size = 12, density = 0.3) {
  const rng = makeRng(seed);
  const obstacles = new Set();
  const start = { r: size - 2, c: 1 };
  const goal = { r: 1, c: size - 2 };
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if ((r === start.r && c === start.c) || (r === goal.r && c === goal.c)) continue;
      if (rng() < density) obstacles.add(`${r},${c}`);
    }
  }
  const visited = new Set();
  const path = [];
  let cur = { ...start };
  let guard = 0;
  while ((cur.r !== goal.r || cur.c !== goal.c) && guard < size * 4) {
    guard++;
    path.push({ ...cur });
    visited.add(`${cur.r},${cur.c}`);
    const options = [];
    if (cur.r > goal.r) options.push({ r: cur.r - 1, c: cur.c });
    if (cur.c < goal.c) options.push({ r: cur.r, c: cur.c + 1 });
    if (rng() < 0.3) options.push({ r: cur.r, c: cur.c - 1 });
    const next = options.find((o) => !obstacles.has(`${o.r},${o.c}`)) || options[0] || cur;
    cur = next;
  }
  path.push(goal);
  return { size, obstacles, start, goal, visited, path };
}

export function curve(n, { from = 15, to = 80, seed = 1, noise = 5 } = {}) {
  const rng = makeRng(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const base = from + (to - from) * (1 - Math.exp(-3.2 * t));
    out.push(Math.max(0, Math.min(100, base + (rng() - 0.5) * noise)));
  }
  return out;
}

/* ---------------- derived datasets ---------------- */
export const episodesAxis = Array.from({ length: 40 }, (_, i) => i * 50);
const successRandom = curve(40, { from: 12, to: 46, seed: 11, noise: 8 });
const successFixed = curve(40, { from: 14, to: 61, seed: 22, noise: 7 });
const successGeneric = curve(40, { from: 13, to: 69, seed: 33, noise: 7 });
const successFailure = curve(40, { from: 16, to: 88, seed: 44, noise: 6 });
export const comparisonData = episodesAxis.map((ep, i) => ({
  ep,
  Random: Math.round(successRandom[i]),
  Fixed: Math.round(successFixed[i]),
  "Generic adaptive": Math.round(successGeneric[i]),
  "Failure driven": Math.round(successFailure[i]),
}));

const lpEpisodes = Array.from({ length: 30 }, (_, i) => i * 60);
const lpSuccess = curve(30, { from: 20, to: 84, seed: 5, noise: 6 });
const lpProgress = lpSuccess.map((v, i, arr) => (i === 0 ? 0 : +(v - arr[i - 1]).toFixed(2)));
export const learningProgressData = lpEpisodes.map((ep, i) => ({
  ep,
  success: Math.round(lpSuccess[i]),
  progress: +(lpProgress[i] / 4).toFixed(2),
}));

export const weaknesses = [
  { key: "long_horizon_planning", label: "Long-horizon planning", value: 0.78, color: C.orange },
  { key: "dead_end_recovery", label: "Dead-end recovery", value: 0.64, color: C.orange },
  { key: "path_efficiency", label: "Path efficiency", value: 0.51, color: C.yellow },
  { key: "exploration", label: "Exploration", value: 0.32, color: C.blue },
  { key: "obstacle_avoidance", label: "Obstacle avoidance", value: 0.18, color: C.green },
];

export const failureLog = [
  { ep: 1839, type: "Long-horizon failure", severity: 0.82, confidence: 0.88 },
  { ep: 1840, type: "Dead-end recovery", severity: 0.65, confidence: 0.74 },
  { ep: 1841, type: "Long-horizon failure", severity: 0.79, confidence: 0.85 },
  { ep: 1842, type: "Path inefficiency", severity: 0.58, confidence: 0.71 },
  { ep: 1843, type: "Dead-end recovery", severity: 0.61, confidence: 0.69 },
  { ep: 1844, type: "Exploration failure", severity: 0.35, confidence: 0.6 },
];

export const curriculumEnvs = [
  { id: "ENV-036", weakness: "Exploration", difficulty: 0.21, success: 0.45, lp: null, status: "done" },
  { id: "ENV-037", weakness: "Exploration", difficulty: 0.34, success: 0.56, lp: 0.11, status: "done" },
  { id: "ENV-038", weakness: "Dead-end recovery", difficulty: 0.48, success: 0.61, lp: 0.05, status: "done" },
  { id: "ENV-039", weakness: "Path efficiency", difficulty: 0.56, success: 0.65, lp: 0.04, status: "done" },
  { id: "ENV-040", weakness: "Long-horizon planning", difficulty: 0.62, success: 0.69, lp: 0.06, status: "done" },
  { id: "ENV-041", weakness: "Long-horizon planning", difficulty: 0.71, success: 0.74, lp: 0.05, status: "done" },
  { id: "ENV-042", weakness: "Long-horizon planning", difficulty: 0.61, success: 0.74, lp: null, status: "current" },
  { id: "ENV-043", weakness: "Long-horizon planning", difficulty: 0.62, success: null, lp: null, status: "pending" },
];

export const candidates = [
  { id: "ENV-043", seed: 501, difficulty: 0.62, alignment: 0.91, novelty: 0.72, validity: 1, selected: true },
  { id: "ENV-044", seed: 502, difficulty: 0.67, alignment: 0.87, novelty: 0.81, validity: 1, selected: false },
  { id: "ENV-045", seed: 503, difficulty: 0.58, alignment: 0.79, novelty: 0.64, validity: 1, selected: false },
  { id: "ENV-046", seed: 504, difficulty: 0.7, alignment: 0.74, novelty: 0.9, validity: 0.92, selected: false },
];

export const experiments = [
  { id: "EXP-024", curriculum: "Failure driven", algo: "PPO", status: "training", success: 0.74, episodes: "1,842 / 10,000", created: "Aug 18, 10:24" },
  { id: "EXP-023", curriculum: "Generic adaptive", algo: "PPO", status: "completed", success: 0.69, episodes: "10,000 / 10,000", created: "Aug 17, 22:05" },
  { id: "EXP-022", curriculum: "Fixed", algo: "PPO", status: "completed", success: 0.61, episodes: "10,000 / 10,000", created: "Aug 17, 14:40" },
  { id: "EXP-021", curriculum: "Random", algo: "PPO", status: "completed", success: 0.46, episodes: "10,000 / 10,000", created: "Aug 16, 09:11" },
  { id: "EXP-020", curriculum: "Failure driven", algo: "PPO", status: "failed", success: 0.28, episodes: "2,140 / 10,000", created: "Aug 15, 19:52" },
];

export const generalization = [
  { split: "Training", success: 88, reward: 91, steps: 24, collision: 4 },
  { split: "Validation", success: 79, reward: 82, steps: 29, collision: 6 },
  { split: "Test (unseen)", success: 71, reward: 74, steps: 33, collision: 9 },
  { split: "OOD", success: 58, reward: 60, steps: 41, collision: 15 },
];
