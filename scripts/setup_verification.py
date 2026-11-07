#!/usr/bin/env python3
"""Install pinned proof tools into .tools; requires network access and rustup/elan."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    pin = json.loads((ROOT / "verification/tools.json").read_text())
    system = platform.system() + "-" + platform.machine()
    if system not in pin["assets"]: raise SystemExit(f"No pinned Aeneas binary for {system}; build the pinned source release separately")
    asset = pin["assets"][system]
    work = ROOT / ".tools"
    work.mkdir(exist_ok=True)
    archive = work / asset["name"]
    if not archive.is_file():
        url = f"https://github.com/AeneasVerif/aeneas/releases/download/{pin['aeneas_tag']}/{asset['name']}"
        urllib.request.urlretrieve(url, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != asset["sha256"]:
        raise SystemExit("Aeneas archive SHA256 mismatch")
    bundle = work / "aeneas"
    bundle.mkdir(exist_ok=True)
    if not (bundle / "aeneas").is_file():
        with tarfile.open(archive) as tar: tar.extractall(bundle, filter="data")
    env = os.environ.copy()
    env.update({"ELAN_HOME": str(work / "elan"), "RUSTUP_HOME": str(work / "rustup"), "CARGO_HOME": str(work / "cargo")})
    commands = [["elan", "toolchain", "install", "leanprover/lean4:v4.24.0"],
                ["elan", "toolchain", "install", pin["aeneas_lean"]],
                ["rustup", "toolchain", "install", pin["charon_rust"], "--profile", "minimal", "--component", "rustc-dev,rust-src"]]
    for command in commands: subprocess.run(command, cwd=ROOT, env=env, check=True)
    for project in [ROOT / "verification", ROOT / "verification/aeneas"]:
        subprocess.run(["lake", "update"], cwd=project, env=env, check=True)
    print("Pinned proof tools installed in .tools")

if __name__ == "__main__": main()
