#!/usr/bin/env python3
"""Generate the complete reference inventory with explicit support overrides."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def run(args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    reference = json.loads(run(["Rscript", "--vanilla", "scripts/inventory.R"]))
    groups = json.loads((ROOT / "verification" / "components.json").read_text())
    components = {}
    for group in groups:
        for name in group["r_functions"]:
            if name in components: raise RuntimeError(f"duplicate support mapping: {name}")
            components[name] = {k: v for k, v in group.items() if k != "r_functions"}
    definitions = reference["definitions"]
    by_name = {d["name"]: d for d in definitions}
    for name, component in components.items():
        if name not in by_name:
            raise RuntimeError(f"support map names no R definition: {name}")
        for file in component["rust_files"]:
            if not (ROOT / file).is_file():
                raise RuntimeError(f"missing Rust implementation {file}")
        for test in component["tests"]:
            if not (ROOT / test).is_file():
                raise RuntimeError(f"missing equivalence test {test}")
    def support(name, seen=None):
        seen = set() if seen is None else seen
        if name in seen: return None
        seen.add(name)
        if name in components: return components[name]
        definition = by_name.get(name)
        if definition and definition["kind"] == "alias": return support(definition["target"], seen)
        return None
    counts = collections.Counter()
    for definition in definitions:
        component = support(definition["name"])
        definition["support"] = component or {"status": "unsupported"}
        counts[definition["support"]["status"]] += 1
    exports = [{"name": name, "support": support(name) or {"status": "unsupported"}} for name in reference.pop("exports")]
    hashes = {}
    source = ROOT / "reference" / "lavaan"
    paths = [source / "DESCRIPTION", source / "NAMESPACE"] + sorted((source / "R").glob("*.R")) + sorted((source / "data").glob("*"))
    tree = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative] = digest
        tree.update((relative + "\0" + digest + "\n").encode())
    report = {"reference_commit": run(["git", "-C", "reference/lavaan", "rev-parse", "HEAD"]).strip(),
              "reference_tree_sha256": tree.hexdigest(), "source_sha256": hashes,
              "summary": {"definitions": len(definitions), "exports": len(exports),
                          "definition_statuses": dict(sorted(counts.items())),
                          "export_statuses": dict(sorted(collections.Counter(e["support"]["status"] for e in exports).items()))},
              "note": "Counts include aliases and do not measure functional completeness. supported-domain means only the documented Rust domain; partial means missing branches/options. All unlisted functionality is unsupported.",
              **reference, "exports": exports}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = ROOT / "reports" / "coverage.json"
    if args.check:
        if not output.is_file() or output.read_text() != text:
            raise SystemExit("inventory is stale: run python3 scripts/inventory.py")
    else:
        output.write_text(text)
    print(json.dumps(report["summary"], indent=2))

if __name__ == "__main__": main()
