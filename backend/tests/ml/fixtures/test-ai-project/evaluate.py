"""Section 96 -- minimal evaluation script used by ai.env's own tests.
Writes `outputs/metrics.json` (Section 24's metrics_file mode) and also
prints the same payload as the last line of stdout (metrics_source=
'stdout' mode), so both collection modes can be exercised against one
fixture."""
import argparse
import json
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()

    payload = {
        "status": "success",
        "metrics": {"accuracy": 0.968, "f1": 0.966},
        "samples": 100,
        "errors": 3,
        "artifact": "outputs/best_model.pt",
    }

    out = pathlib.Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload))

    print(f"evaluated on dataset={args.dataset!r}")
    print(json.dumps(payload))
