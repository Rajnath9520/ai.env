"""Section 96 -- a minimal stand-in for a real training script, used only
by ai.env's own CommandAdapter/scanner tests. Writes a placeholder
artifact file so `artifact.path` checks have something real to find."""
import argparse
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()

    out = pathlib.Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "best_model.pt").write_bytes(b"not-a-real-checkpoint")
    print(f"trained on dataset={args.dataset!r}")
