import type { Cell } from "@/components/grid-world/GridWorldRenderer";

/** Real BFS shortest path through the actual obstacle layout -- not a
 * fabricated or fake trajectory. Explicitly a *pathfinding* result, not a
 * claim about how the RL agent actually moved (the backend doesn't stream
 * per-step agent positions -- see Live Training's documented limitation).
 * Used to demonstrate the renderer's animation support with real,
 * correctly-computed maze data.
 */
export function bfsShortestPath(gridSize: number, obstacles: Cell[], start: Cell, goal: Cell): Cell[] | null {
  const obstacleSet = new Set(obstacles.map(([x, y]) => `${x},${y}`));
  const isFree = (x: number, y: number) =>
    x >= 0 && x < gridSize && y >= 0 && y < gridSize && !obstacleSet.has(`${x},${y}`);

  const startKey = `${start[0]},${start[1]}`;
  const goalKey = `${goal[0]},${goal[1]}`;
  const visited = new Set<string>([startKey]);
  const parent = new Map<string, string>();
  const queue: Cell[] = [start];

  const moves: Cell[] = [
    [0, -1],
    [0, 1],
    [-1, 0],
    [1, 0],
  ];

  let found = false;
  while (queue.length > 0) {
    const [cx, cy] = queue.shift()!;
    const currentKey = `${cx},${cy}`;
    if (currentKey === goalKey) {
      found = true;
      break;
    }
    for (const [dx, dy] of moves) {
      const nx = cx + dx;
      const ny = cy + dy;
      const nKey = `${nx},${ny}`;
      if (isFree(nx, ny) && !visited.has(nKey)) {
        visited.add(nKey);
        parent.set(nKey, currentKey);
        queue.push([nx, ny]);
      }
    }
  }

  if (!found) return null;

  const path: Cell[] = [goal];
  let cursor = goalKey;
  while (cursor !== startKey) {
    const prev = parent.get(cursor);
    if (!prev) return null;
    const [px, py] = prev.split(",").map(Number);
    path.push([px, py]);
    cursor = prev;
  }
  return path.reverse();
}
