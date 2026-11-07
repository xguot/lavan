#!/usr/bin/env python3
"""Extract the same src/indexing.rs compiled by lavan, then check generated Lean."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def run(args, cwd, env):
    subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if committed extraction differs")
    args = parser.parse_args()
    pin = json.loads((ROOT / "verification/tools.json").read_text())
    env = os.environ.copy()
    # Use workspace-local caches when provisioned; otherwise normal tool homes.
    for variable, folder in [("ELAN_HOME", "elan"), ("RUSTUP_HOME", "rustup"), ("CARGO_HOME", "cargo")]:
        local = ROOT / ".tools" / folder
        if local.is_dir(): env.setdefault(variable, str(local))
    env["RUSTUP_TOOLCHAIN"] = pin["charon_rust"]
    rustup = shutil.which("rustup", path=env.get("PATH"))
    if not rustup: raise SystemExit("rustup is required")
    rustc = subprocess.check_output([rustup, "which", "--toolchain", pin["charon_rust"], "rustc"], env=env, text=True).strip()
    env["PATH"] = str(Path(rustc).parent) + os.pathsep + env.get("PATH", "")
    bundle = ROOT / ".tools" / "aeneas"
    if not (bundle / "aeneas").is_file(): raise SystemExit("Run python3 scripts/setup_verification.py first")
    version = subprocess.check_output([str(bundle / "aeneas"), "-version"], text=True).strip()
    if version != "aeneas " + pin["aeneas_tag"]: raise SystemExit("Aeneas version differs from verification/tools.json")
    project = ROOT / "verification" / "aeneas"
    with tempfile.TemporaryDirectory(prefix="lavan-extract-", dir=ROOT / ".tools") as temporary:
        temporary = Path(temporary)
        llbc = temporary / "lavan-kernels.llbc"
        run([bundle / "charon", "cargo", "--preset=aeneas", "--sysroot", "default", "--dest-file", llbc, "--", "--lib"], ROOT / "verification/rust-kernels", env)
        run([bundle / "aeneas", "-backend", "lean", "-dest", temporary, "-namespace", "lavan_kernels", "-use-lean-modules", "false", "-abort-on-error", "-no-progress-bar", llbc], ROOT, env)
        generated = (temporary / "Lavan-kernels.lean").read_text()
        committed = project / "LavanKernels.lean"
        if args.check:
            if not committed.is_file() or committed.read_text() != generated:
                raise SystemExit("Rust extraction changed: run python3 scripts/extract_rust.py")
        else:
            committed.write_text(generated)
    run(["lake", "build"], project, env)
    print("Rust extraction and Lean proofs passed; source SHA256:", hashlib.sha256((ROOT / "src/indexing.rs").read_bytes()).hexdigest())

if __name__ == "__main__": main()
