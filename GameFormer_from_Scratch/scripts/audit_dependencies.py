"""Audit exact lock selections; a skipped advisory query fails the gate."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_report(report):
    dependencies = report.get("dependencies", [])
    if not dependencies:
        raise RuntimeError("No dependencies were audited")
    skipped = {
        item["name"]: item["skip_reason"] for item in dependencies if item.get("skip_reason")
    }
    if skipped:
        raise RuntimeError(f"Dependencies remain unaudited: {skipped}")
    vulnerable = {item["name"]: item["vulns"] for item in dependencies if item.get("vulns")}
    if vulnerable:
        raise RuntimeError(f"Known vulnerabilities: {vulnerable}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/checks/dependency-audit.json")
    args = parser.parse_args()
    directory = ROOT / "outputs/checks"
    directory.mkdir(parents=True, exist_ok=True)
    locked = directory / "audit-locked-requirements.txt"
    subprocess.run(["uv", "lock", "--check"], cwd=ROOT, check=True)
    subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--all-extras",
            "--no-emit-project",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--quiet",
            "--output-file",
            str(locked),
        ],
        cwd=ROOT,
        check=True,
    )
    # Advisory databases index the public release (2.x.y), not +cpu wheel tags.
    # This changes only the advisory query; installation still uses hashed uv.lock.
    query = directory / "audit-query-requirements.txt"
    query.write_text(re.sub(r"(?m)^(torch==[0-9][^\s;+]*)\+cpu(?=\s|$)", r"\1", locked.read_text()))
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--no-deps",
            "--disable-pip",
            "--progress-spinner",
            "off",
            "-r",
            str(query),
            "--format",
            "json",
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        check=False,
    )
    report = json.loads(destination.read_text())
    report["audit_provenance"] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "query_protocol": "All extras exported from frozen uv.lock; torch +cpu suffix queried as its public release; no packages skipped or CVEs ignored",
        "python": sys.version.split()[0],
        "lock_selection": "PEP 508 markers evaluated by pip-audit on this platform/interpreter",
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    validate_report(report)
    completed.check_returncode()
    print(f"Audited {len(report['dependencies'])} dependencies with no skipped packages")


if __name__ == "__main__":
    main()
