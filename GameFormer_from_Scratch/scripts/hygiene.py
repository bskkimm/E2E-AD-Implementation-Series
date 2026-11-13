"""Fast file-integrity checks, also used as the local pre-commit hook."""

import json
import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def check(path):
    failures = []
    # The generated multi-platform lock is allowed 2 MiB; all other files 1 MiB.
    limit = 2 * 1024**2 if path.name == "uv.lock" else 1024**2
    raw = path.read_bytes()
    if len(raw) > limit:
        failures.append("file exceeds size limit")
    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif"):
        return failures
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return failures + ["unexpected binary file"]
    if raw and (not text.endswith("\n") or text.endswith("\n\n")):
        failures.append("exactly one final newline required")
    if any(line != line.rstrip() for line in text.splitlines()):
        failures.append("trailing whitespace")
    if re.search(r"^(?:<{7}|>{7}|={7})", text, re.M):
        failures.append("merge conflict marker")
    if re.search(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----", text):
        failures.append("private key")
    try:
        if path.suffix in (".json", ".ipynb"):
            value = json.loads(text)
            if path.suffix == ".ipynb":
                if any(
                    c.get("outputs") or c.get("execution_count") is not None
                    for c in value["cells"]
                    if c["cell_type"] == "code"
                ):
                    failures.append("strip notebook outputs/execution counts before committing")
        elif path.suffix in (".yaml", ".yml"):
            yaml.safe_load(text)
        elif path.suffix == ".toml":
            tomllib.loads(text)
    except (ValueError, yaml.YAMLError) as error:
        failures.append(str(error))
    return failures


def main():
    failed = {}
    for name in sys.argv[1:]:
        path = Path(name).resolve()
        if path.is_file():
            errors = check(path)
            if errors:
                failed[name] = errors
    if failed:
        raise SystemExit(json.dumps(failed, indent=2))
    print(f"File hygiene passed ({len(sys.argv) - 1} files)")


if __name__ == "__main__":
    main()
