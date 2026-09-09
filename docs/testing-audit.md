# Phase 26 — Testing Audit

This phase's job is auditing and consolidating test coverage already built
incrementally across every prior phase, filling real gaps (frontend unit
tests) rather than re-writing what already exists. **312 backend tests +
8 frontend tests = 320 tests total**, all passing.

## Section 42's 10 required critical tests -- mapped to real coverage

| # | Requirement | Test file(s) | Count |
|---|---|---|---|
| 1 | Environment validity | `tests/ml/test_gridworld.py` (generator/validator), plus a manual 540-config stress sweep (Phase 1) | 55 |
| 2 | Agent training | `tests/ml/test_trainer.py` | 7 |
| 3 | Trajectory recording | `tests/ml/test_trajectory.py` | 11 |
| 4 | Failure detection | `tests/ml/test_failure_analyzer.py`, `tests/ml/test_dead_ends.py` | 16 |
| 5 | Weakness generation | `tests/ml/test_weakness_vector.py` | 11 |
| 6 | Candidate environment generation | `tests/ml/test_candidate_generator.py`, `tests/ml/test_synthesizer.py` | 22 |
| 7 | Learning progress calculation | `tests/ml/test_learning_progress.py` | 10 |
| 8 | Curriculum selection | `tests/ml/test_curriculum_controller.py`, `test_random_curriculum.py`, `test_fixed_curriculum.py`, `test_generic_adaptive_curriculum.py`, `test_failure_driven.py`, `test_ablation.py` | 63 |
| 9 | Generalization separation | `tests/ml/test_generalization.py` | 11 |
| 10 | API functionality | `tests/api/*.py` (CRUD, lifecycle, WebSocket, broadcaster, environment layout) | 32 |

Every item on Section 42's critical list has dedicated, passing test
coverage -- not incidental coverage picked up as a side effect of testing
something else.

## Coverage by test category (Section 42's broader list)

- **Unit tests**: the large majority of the 312 backend tests -- pure-logic
  tests against hand-built inputs (e.g. `test_metrics.py`'s exact-arithmetic
  checks, `test_dead_ends.py`'s hand-built maze graphs).
- **Integration tests**: `tests/ml/test_experiment_manager.py` (full
  `ExperimentManager.run()` for all 4 curriculum types end-to-end, including
  generalization), `tests/ml/test_reproducibility.py`'s `reproduce()` test
  (real re-run from a saved manifest, diffed against the original),
  `tests/db/test_repository.py` (real `ExperimentManager` run persisted
  into real PostgreSQL and queried back).
- **API tests**: `tests/api/*.py`, 32 tests against a real FastAPI
  `TestClient` and real PostgreSQL -- CRUD, status-transition edge cases
  (409 on double-start, 404s), WebSocket streaming, layout endpoint.
- **Environment tests**: `tests/ml/test_gridworld.py`,
  `tests/ml/test_dead_ends.py` -- generator determinism, validity guarantees,
  Gymnasium contract compliance.
- **ML pipeline tests**: everything under `tests/ml/` not already listed
  above -- trainer, trajectory, failure analysis, weakness vectors,
  synthesis, candidates, curricula, learning progress, generalization,
  metrics, statistics, ablation, reproducibility.
- **Frontend tests**: added this phase (see below) -- 8 tests via Vitest,
  covering the pure-logic pieces that are practical to unit test without a
  full browser/DOM harness.

## Frontend tests (new this phase)

Section 42: "Frontend tests where practical." A full component-level test
harness (jsdom + React Testing Library) was judged disproportionate to
this phase's scope for a dashboard that is primarily data display wired to
a well-tested backend; instead, the frontend's non-trivial *pure logic* --
where a real bug is actually plausible and a unit test is cheap and
fast -- got real Vitest coverage:

- `lib/grid-pathfinding.test.ts` (5 tests) -- BFS shortest-path correctness:
  finds the direct route on an open grid, never steps on an obstacle,
  correctly returns `null` when the goal is walled off, handles the
  start-equals-goal edge case, and only ever takes orthogonal steps.
- `lib/api-client.test.ts` (3 tests) -- `wsUrlForExperiment` URL
  construction and `ApiError`'s status/body handling.

Run via `npm test` (or `npx vitest run`) in `project-root/frontend`.
`npm run build` and `npm run lint` remain the primary correctness signal
for the React components themselves (both clean, per Phases 19-20's
reports), since TypeScript's type checker already catches the class of
bugs component-level tests would otherwise need to.

## What was NOT added, and why

- No formal code-coverage percentage tool (e.g. `pytest-cov`) was wired
  in -- the audit above is a direct requirement-to-test mapping, which is
  more meaningful than a raw line-coverage number for judging whether the
  *required* things are tested, though adding `pytest-cov` would be a
  reasonable follow-up for ongoing development.
- No browser/E2E tests (Playwright/Cypress) for the frontend -- judged out
  of scope for "where practical" given the added infrastructure (headless
  browser, test fixtures) relative to the benefit at this project's
  current stage; the backend's own extensive API tests already exercise
  every code path the frontend calls into.
