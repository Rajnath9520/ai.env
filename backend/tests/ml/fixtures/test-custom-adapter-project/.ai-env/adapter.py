"""Fixture custom adapter (Section 11) used by CustomAdapter's own tests.
Implements all three optional methods so capability discovery, train,
evaluate, and predict can all be exercised against one well-behaved
adapter; tests that need a partial/broken adapter write their own into a
tmp_path fixture instead of modifying this one."""
from pathlib import Path


class FixtureAdapter:
    def train(self, config):
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        (out / "model.bin").write_text("trained-by-custom-adapter")
        return {"trained": True, "dataset": config.get("variables", {}).get("dataset")}

    def evaluate(self, config):
        return {"status": "success", "metrics": {"accuracy": 0.87, "f1": 0.83}, "samples": 50}

    def predict(self, config):
        return {"echo": config.get("input")}


def get_adapter():
    return FixtureAdapter()
