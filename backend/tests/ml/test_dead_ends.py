from app.ml.analysis.dead_ends import dead_end_zone_cells, reachable_cells
from app.ml.environments.validator import GridWorldLayout


class TestDeadEndZone:
    def test_single_leaf_spur_is_a_dead_end_zone(self):
        # 5x5 open room from (0,0)-(2,2), with a spur (3,1)-(4,1) hanging
        # off (2,1) that leads nowhere.
        n = 5
        all_cells = {(x, y) for x in range(n) for y in range(n)}
        keep = {(x, y) for x in range(3) for y in range(3)} | {(3, 1), (4, 1)}
        obstacles = all_cells - keep
        layout = GridWorldLayout(
            grid_size=n, agent_start=(0, 0), goal=(2, 2), obstacles=frozenset(obstacles)
        )
        reachable = reachable_cells(layout)
        zone = dead_end_zone_cells(layout, reachable)
        assert (3, 1) in zone
        assert (4, 1) in zone
        # The open room itself (including start/goal) is not a dead end.
        assert (1, 1) not in zone
        assert layout.agent_start not in zone
        assert layout.goal not in zone

    def test_loop_is_not_a_dead_end_zone(self):
        # A 3x3 ring (perimeter of a 3x3 block) has no degree-1 cells at all —
        # every cell in the loop has >= 2 neighbors, so nothing gets pruned.
        n = 5
        ring = {(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)}
        all_cells = {(x, y) for x in range(n) for y in range(n)}
        obstacles = all_cells - ring
        layout = GridWorldLayout(
            grid_size=n, agent_start=(1, 1), goal=(3, 3), obstacles=frozenset(obstacles)
        )
        zone = dead_end_zone_cells(layout, reachable_cells(layout))
        assert zone == frozenset()

    def test_branch_with_alternate_loop_route_is_not_pruned(self):
        # Two parallel corridors both connecting (0,0) to (0,2) form a loop;
        # neither corridor should be treated as a dead end even though each
        # individually "looks like" a spur.
        n = 4
        loop = {(0, 0), (1, 0), (1, 1), (1, 2), (0, 2), (0, 1)}
        all_cells = {(x, y) for x in range(n) for y in range(n)}
        obstacles = all_cells - loop
        layout = GridWorldLayout(
            grid_size=n, agent_start=(0, 0), goal=(0, 2), obstacles=frozenset(obstacles)
        )
        zone = dead_end_zone_cells(layout, reachable_cells(layout))
        assert zone == frozenset()
