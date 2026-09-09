from scripts.run_research_experiments import run_one, summarize
import pytest


class TestRunOne:
    def test_run_one_returns_well_formed_entry_and_saves_file(self, tmp_path, monkeypatch):
        import scripts.run_research_experiments as mod

        monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(
            mod,
            "SHARED_KWARGS",
            dict(
                agent_mode="demo_ppo",
                grid_size=5,
                obstacle_density=0.1,
                dead_end_probability=0.1,
                path_complexity=0.1,
                max_steps_per_episode=12,
                n_curriculum_steps=2,
                timesteps_per_step=256,
                eval_episodes=3,
                n_candidates=2,
                trajectories_per_analysis=2,
                run_generalization=True,
                generalization_envs_per_split=2,
                generalization_episodes_per_config=2,
            ),
        )

        summary = {}
        entry = run_one("random", 1, summary)

        assert entry["seed"] == 1
        assert 0.0 <= entry["final_success_rate"] <= 1.0
        assert entry["generalization"] is not None
        assert set(entry["generalization"].keys()) == {"validation", "test", "ood"}
        assert summary["random"][0] is entry
        assert (tmp_path / "random_seed1.json").exists()


class TestSummarize:
    def test_summarize_computes_means_correctly(self):
        summary = {
            "random": [
                {
                    "seed": 1,
                    "final_success_rate": 0.2,
                    "generalization": {
                        "validation": {"success_rate": 0.1},
                        "test": {"success_rate": 0.15},
                        "ood": {"success_rate": 0.1},
                    },
                },
                {
                    "seed": 2,
                    "final_success_rate": 0.4,
                    "generalization": {
                        "validation": {"success_rate": 0.3},
                        "test": {"success_rate": 0.25},
                        "ood": {"success_rate": 0.3},
                    },
                },
            ]
        }
        report = summarize(summary)
        assert report["random"]["final_success_rate_mean"] == pytest.approx(0.3)
        assert report["random"]["ood_success_rate_mean"] == pytest.approx(0.2)
        assert report["random"]["validation_success_rate_mean"] == pytest.approx(0.2)
        assert report["random"]["test_success_rate_mean"] == pytest.approx(0.2)
        assert report["random"]["seeds"] == [1, 2]

    def test_summarize_handles_missing_generalization(self):
        summary = {"fixed": [{"seed": 1, "final_success_rate": 0.5, "generalization": None}]}
        report = summarize(summary)
        assert report["fixed"]["ood_success_rate_per_seed"] == []
        assert report["fixed"]["ood_success_rate_mean"] is None
