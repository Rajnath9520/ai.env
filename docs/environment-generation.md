# Environment Generation

## Configuration

`GridWorldConfig` (`app/ml/environments/config.py`) is the single source
of truth for a concrete environment's parameters:

| Field | Range | Meaning |
|---|---|---|
| `grid_size` | 4-24 | width/height of the square grid |
| `obstacle_density` | 0.0-0.9 | fraction of non-essential cells that become obstacles |
| `goal_distance` | int or `None` | minimum Manhattan distance between start and goal |
| `dead_end_probability` | 0.0-1.0 | weight controlling extra dead-end branch carving |
| `path_complexity` | 0.0-1.0 | weight controlling extra loop-branch carving (alternate routes) |
| `max_steps` | >=1 | episode step limit |
| `seed` | int or `None` | fully determines the generated layout |

## Generation algorithm (guarantee-by-construction)

`GridWorldGenerator.generate()` never uses reject-and-retry. Instead:

1. Sample start/goal positions (respecting `goal_distance` if set).
2. Carve a biased random-walk path connecting them (falls back to a
   deterministic L-shaped path if the walk doesn't converge quickly) --
   every cell on this path is *protected* (always free).
3. Carve extra branches off the path: loop branches (reconnecting, adding
   alternate routes) weighted by `path_complexity`, and dead-end spurs
   weighted by `dead_end_probability`.
4. Scatter obstacles over whatever cells remain, at `obstacle_density`.
5. Validate (BFS reachability) as a confirmation check -- this should
   always pass by construction; if it doesn't, that's treated as a
   generator bug, not an expected outcome, and the generator raises rather
   than silently returning an invalid layout.

This was stress-tested across 540 parameter combinations (grid sizes
4-25, full density/complexity range) with **zero invalid environments**.

## Targeted synthesis

`EnvironmentSynthesizer` (`app/ml/environments/synthesizer.py`) is the
core proposed-method component: given a `WeaknessVector` and a base
config, it adjusts *only* the parameters causally linked to the targeted
weakness, scaled by that weakness's measured severity:

| Weakness | Parameters moved |
|---|---|
| `obstacle_avoidance` | obstacle_density only |
| `dead_end_recovery` | dead_end_probability, path_complexity |
| `exploration` | grid_size, goal_distance |
| `path_efficiency` | path_complexity only |
| `short_horizon_planning` | obstacle_density, path_complexity |
| `long_horizon_planning` | goal_distance, path_complexity, dead_end_probability |

Every other parameter is left exactly as in the base config -- this is
what makes it "targeted" rather than "generically harder," and is
verified directly in tests (e.g. targeting `obstacle_avoidance` never
changes `grid_size`).

## Candidate generation

`CandidateGenerator` (`app/ml/environments/candidate_generator.py`)
produces several candidates per curriculum step rather than committing to
one: it cycles targets through `[primary_weakness, *secondary_weaknesses]`,
jitters severity per candidate, and scores each on:

- `difficulty` -- the same heuristic estimate used everywhere else
- `validity_score` -- 1.0/0.0 from actually running the candidate through
  the real generator+validator, not assumed
- `diversity_score` -- normalized 5-D Euclidean distance in parameter
  space against prior environments *and* prior candidates in the same
  batch

## Known limitations

- Mapping weights (`WEAKNESS_TARGET_PARAMS`) are hand-specified and
  documented, not empirically calibrated against real learning-progress
  outcomes.
- `grid_size` is only ever adjusted for `exploration` -- every other
  weakness keeps it fixed, to avoid every synthesized environment quietly
  becoming "the same thing but bigger."
