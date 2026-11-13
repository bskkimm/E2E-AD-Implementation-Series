"""Check local links, named recipes, CLI coverage and recorded evidence hashes."""

import hashlib
import json
import re
from pathlib import Path

from gameformer.config import RunConfig, recipe

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in (
        "README.md",
        "COMMAND_GUIDE.md",
        "docs/setup.md",
        "docs/data.md",
        "docs/results.md",
        "docs/benchmark.md",
        "docs/visual-inspection.md",
        "docs/ci-review.md",
        "docs/paper-to-code.md",
        "docs/results-manifest.json",
    ):
        assert (ROOT / name).is_file(), f"Missing deliverable: {name}"
    for path in ROOT.rglob("*.md"):
        if any(
            part.startswith(".") or part in ("outputs", "data", "build")
            for part in path.relative_to(ROOT).parts
        ):
            continue
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if re.match(r"(?:https?://|mailto:|#)", target):
                continue
            assert (path.parent / target.split("#")[0]).exists(), f"{path}: missing {target}"
    for path in (ROOT / "configs").glob("*.json"):
        config = RunConfig.load(path)
        assert config == recipe(path.stem), f"Named recipe drift: {path}"
    guide = (ROOT / "COMMAND_GUIDE.md").read_text()
    for command in ("train", "eval", "prepare", "benchmark", "tune"):
        assert f"gameformer {command}" in guide
    manifest = json.loads((ROOT / "docs/results-manifest.json").read_text())
    assert manifest["paper_results_reproduced"] is False
    source_files = [
        *sorted((ROOT / "gameformer").glob("*.py")),
        *sorted((ROOT / "configs").glob("*.json")),
    ]
    current_source = hashlib.sha256(
        "".join(
            f"{path.relative_to(ROOT)}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
            for path in source_files
        ).encode()
    ).hexdigest()
    assert current_source == manifest["smoke_run"]["source_sha256"], "Evidence source is stale"
    resolved = RunConfig.from_dict(manifest["smoke_run"]["config"]).to_dict()
    assert (
        hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
        == manifest["smoke_run"]["config_sha256"]
    )
    workflow = ROOT.parent / ".github/workflows/gameformer.yml"
    assert workflow.read_bytes() == (ROOT / "ci/gameformer.yml").read_bytes()
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"], path
    for field in (
        "code_commit",
        "source_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "seed",
        "dataset",
        "split",
        "metric_protocol",
        "checkpoint_path",
    ):
        assert manifest["smoke_run"].get(field) is not None, field
    assert (
        len(
            json.loads((ROOT / "notebooks/assets/architecture/manifest.json").read_text())[
                "figures"
            ]
        )
        == 18
    )
    print("Documentation, configs, links and evidence hashes passed")


if __name__ == "__main__":
    main()
