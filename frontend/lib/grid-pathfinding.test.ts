import { describe, expect, it } from "vitest";
import { bfsShortestPath } from "@/lib/grid-pathfinding";
import type { Cell } from "@/components/grid-world/GridWorldRenderer";

describe("bfsShortestPath", () => {
  it("finds the direct path on an open grid", () => {
    const path = bfsShortestPath(5, [], [0, 0], [4, 0]);
    expect(path).not.toBeNull();
    expect(path![0]).toEqual([0, 0]);
    expect(path![path!.length - 1]).toEqual([4, 0]);
    expect(path!.length).toBe(5);
  });

  it("returns a path that never steps onto an obstacle", () => {
    const obstacles: Cell[] = [
      [1, 0],
      [1, 1],
      [1, 2],
    ];
    const path = bfsShortestPath(4, obstacles, [0, 0], [3, 0]);
    expect(path).not.toBeNull();
    const obstacleSet = new Set(obstacles.map(([x, y]) => `${x},${y}`));
    for (const [x, y] of path!) {
      expect(obstacleSet.has(`${x},${y}`)).toBe(false);
    }
  });

  it("returns null when the goal is completely walled off", () => {
    const gridSize = 4;
    const obstacles: Cell[] = Array.from({ length: gridSize }, (_, y) => [2, y] as Cell);
    const path = bfsShortestPath(gridSize, obstacles, [0, 0], [3, 3]);
    expect(path).toBeNull();
  });

  it("returns a single-cell path when start equals goal", () => {
    const path = bfsShortestPath(5, [], [2, 2], [2, 2]);
    expect(path).toEqual([[2, 2]]);
  });

  it("path only moves to orthogonally adjacent cells (no diagonal jumps)", () => {
    const path = bfsShortestPath(6, [], [0, 0], [3, 3]);
    expect(path).not.toBeNull();
    for (let i = 1; i < path!.length; i++) {
      const [px, py] = path![i - 1];
      const [cx, cy] = path![i];
      const manhattanStep = Math.abs(px - cx) + Math.abs(py - cy);
      expect(manhattanStep).toBe(1);
    }
  });
});
