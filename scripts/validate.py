#!/usr/bin/env python3
"""Run all current validation gates and record evidence only after success."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def source_hashes():
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    extensions = {".rs", ".R", ".py", ".lean", ".toml", ".json"}
    for directory in ["src", "tests", "benches", "examples", "scripts", "verification"]:
        for current, dirs, files in os.walk(ROOT / directory):
            dirs[:] = [d for d in dirs if d not in {".lake", "target", "__pycache__", ".git"}]
            for name in files:
                path = Path(current) / name
                if path.name == "Lavan-kernels.lean": continue
                if path.suffix in extensions or path.name == "lean-toolchain": paths.append(path)
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}

def main():
    env = os.environ.copy()
    for variable, folder in [("ELAN_HOME", "elan"), ("CARGO_HOME", "cargo")]:
        local = ROOT / ".tools" / folder
        if local.is_dir(): env.setdefault(variable, str(local))
    logs = ROOT / "target" / "validation"
    logs.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "platform": platform.platform(), "machine": platform.machine(),
              "source_sha256": source_hashes(), "checks": []}
    outputs = {}
    def check(name, command, cwd=ROOT):
        print(f"Checking {name}...", flush=True)
        start = time.monotonic()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout
        outputs[name] = output
        log = logs / (name + ".log")
        log.write_text(output)
        report["checks"].append({"name": name, "command": command, "directory": str(cwd.relative_to(ROOT)),
                                 "exit_code": result.returncode, "seconds": time.monotonic() - start,
                                 "log": str(log.relative_to(ROOT))})
        if result.returncode:
            print(output, flush=True)
            raise RuntimeError(f"{name} failed")
        return output
    try:
        report["rustc"] = check("rust-version", ["rustc", "--version"]).strip()
        check("format", ["cargo", "fmt", "--all", "--", "--check"])
        check("clippy", ["cargo", "clippy", "--locked", "--all-targets", "--", "-D", "warnings"])
        check("rust-docs", ["cargo", "doc", "--locked", "--no-deps"])
        check("rust-tests", ["cargo", "test", "--locked", "--", "--nocapture"])
        check("rust-release-tests", ["cargo", "test", "--locked", "--release", "--", "--nocapture"])
        check("example", ["cargo", "run", "--locked", "--release", "--example", "cfa"])
        check("inventory", ["python3", "scripts/inventory.py", "--check"])
        check("mathlib-proofs", ["lake", "build"], ROOT / "verification")
        math_audit = check("mathlib-audit", ["lake", "env", "lean", "Audit.lean"], ROOT / "verification")
        check("rust-extraction-proofs", ["python3", "scripts/extract_rust.py", "--check"])
        rust_audit = check("rust-proof-audit", ["lake", "env", "lean", "Audit.lean"], ROOT / "verification/aeneas")
        allowed = {"propext", "Classical.choice", "Quot.sound"}
        audit = math_audit + rust_audit
        groups = re.findall(r"depends on axioms:\s*\[([^\]]*)\]", audit)
        if len(groups) != 17: raise RuntimeError(f"expected 17 audited theorem results, got {len(groups)}")
        for group in groups:
            axioms = {s.strip() for s in group.split(",") if s.strip()}
            if not axioms <= allowed: raise RuntimeError(f"unapproved proof axioms: {axioms - allowed}")
        report["proofs"] = {"mathematical_theorems": 14, "extracted_rust_theorems": 3,
                             "axioms": sorted(allowed), "audit": audit.strip()}
        match = re.search(r"R equivalence: (\d+) scalar comparisons passed; max abs error ([\deE.+-]+); max scaled error ([\deE.+-]+)", outputs["rust-tests"])
        if not match: raise RuntimeError("missing numerical equivalence evidence")
        report["equivalence"] = {"scalar_comparisons": int(match[1]), "max_absolute_error": float(match[2]),
                                  "max_scaled_error": float(match[3]), "kernel_tolerance": "3e-10 * max(1, abs(R))",
                                  "worst_cases": re.findall(r"Worst (?:absolute|scaled) comparison: (.+)", outputs["rust-tests"]),
                                  "numeric_model_cases": 6, "cfa_frontend_cases": 6}
        report["reference"] = {k: v for k,v in json.loads((ROOT / "reports/coverage.json").read_text()).items() if k in {"version", "r_files", "r_lines", "reference_commit", "reference_tree_sha256", "summary"}}
        if report["source_sha256"] != source_hashes(): raise RuntimeError("source changed during validation")
        report["status"] = "passed"
    except (RuntimeError, OSError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (ROOT / "reports/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("All validation gates passed; reports/validation.json written", flush=True)

if __name__ == "__main__": main()
