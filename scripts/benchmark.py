#!/usr/bin/env python3
"""Compare kernels on identical inputs; setup and process startup excluded."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from validate import source_hashes

ROOT = Path(__file__).resolve().parents[1]

def matrix(rows, cols, f):
    return {"rows": rows, "cols": cols, "data": [f(i, j) for j in range(cols) for i in range(rows)]}

def run(args):
    result = subprocess.run(args, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE)
    return result.stdout

def main():
    work = ROOT / "target" / "benchmark"
    work.mkdir(parents=True, exist_ok=True)
    fixture = {
        "y": matrix(2000, 12, lambda i, j: math.sin((i + 1) * (j + 1) * 0.017) + math.cos((i + 3) * (j + 2) * 0.031)),
        "sigma": matrix(12, 12, lambda i, j: math.exp(-abs(i - j) / 4) + (0.5 if i == j else 0.0)),
        "mu": [math.sin(j * 0.4) for j in range(12)],
        "b": matrix(400, 400, lambda i, j: math.sin((i + 1) * (j + 2) * 0.003)),
        "lambda": matrix(12, 3, lambda i, j: 0.8 if i // 4 == j else 0.0),
        "psi": matrix(3, 3, lambda i, j: 1.0 if i == j else 0.2),
        "theta": matrix(12, 12, lambda i, j: 0.4 if i == j else 0.0),
        "beta": matrix(3, 3, lambda i, j: 0.2 if i == j + 1 else 0.0),
    }
    input_path = work / "input.json"
    input_path.write_text(json.dumps(fixture, separators=(",", ":")))
    rust = json.loads(run(["cargo", "bench", "--locked", "--bench", "kernels", "--", str(input_path)]))
    r = json.loads(run(["Rscript", "--vanilla", "tests/r/benchmark.R", str(input_path)]))
    rows = []
    for a, b in zip(rust["results"], r["results"], strict=True):
        assert a["name"] == b["name"]
        if not math.isclose(a["checksum"], b["checksum"], rel_tol=3e-10, abs_tol=3e-10):
            raise RuntimeError(f"benchmark output mismatch for {a['name']}")
        row = {"name": a["name"], "rust_median_us": a["median_us"], "r_median_us": b["median_us"], "r_over_rust": b["median_us"] / a["median_us"]}
        rows.append(row)
        print(f"{row['name']}: Rust {row['rust_median_us']:.2f} us, R {row['r_median_us']:.2f} us, R/Rust {row['r_over_rust']:.2f}")
    report = {"measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_sha256": source_hashes(),
              "platform": platform.platform(), "machine": platform.machine(),
              "rustc": run(["rustc", "--version"]).strip(),
              "cargo_lock_sha256": hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest(),
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "reference_commit": run(["git", "-C", "reference/lavaan", "rev-parse", "HEAD"]).strip(),
              "method": "Same serialized inputs, three warmups, >=60ms calibration, seven samples, median wall time; process startup/input parsing excluded; release Rust. Single machine, no statistical speed claim.",
              "dimensions": {"data": [2000, 12], "duplication_input": [400, 400], "lisrel": [12, 3]},
              "summary": rows, "rust": rust, "r": r}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "benchmarks.json").write_text(json.dumps(report, indent=2) + "\n")

if __name__ == "__main__":
    main()
