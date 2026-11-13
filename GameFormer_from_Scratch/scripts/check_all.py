"""Independent CI stages with retained logs; run from the project directory."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "pre-commit",
    "lint-format",
    "types",
    "unit",
    "integration",
    "notebook",
    "package",
    "dependencies",
    "docs-config",
)


def files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "."],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [str(ROOT / path) for path in result.stdout.splitlines() if (ROOT / path).is_file()]


def run(stage, timeout):
    logs = ROOT / "outputs/checks"
    logs.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", str(logs / "matplotlib")),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    env.pop("PYTHONPATH", None)
    env.setdefault("GOPATH", "/tmp/gameformer-go")
    env.setdefault("GOCACHE", "/tmp/gameformer-go-build")
    commands = []
    py = sys.executable
    if stage == "pre-commit":
        root = ROOT.parent
        selected = [os.path.relpath(path, root) for path in files()]
        workflow = root / ".github/workflows/gameformer.yml"
        if workflow.is_file():
            selected.append(str(workflow.relative_to(root)))
        commands.append(
            (
                [
                    py,
                    "-m",
                    "pre_commit",
                    "run",
                    "--config",
                    str(ROOT / ".pre-commit-config.yaml"),
                    "--files",
                    *selected,
                ],
                root,
                env,
            )
        )
    elif stage == "lint-format":
        for options in (["check"], ["format", "--check"]):
            commands.append(
                (
                    [py, "-m", "ruff", *options, "gameformer", "scripts", "tests", "notebooks"],
                    ROOT,
                    env,
                )
            )
        workflow = ROOT.parent / ".github/workflows/gameformer.yml"
        if workflow.is_file():
            commands.append(
                (
                    [
                        "go",
                        "run",
                        "github.com/rhysd/actionlint/cmd/actionlint@v1.7.7",
                        str(workflow),
                    ],
                    ROOT,
                    env,
                )
            )
    elif stage == "types":
        commands.append(([py, "-m", "mypy"], ROOT, env))
    elif stage in ("unit", "integration", "notebook"):
        tests = {
            "unit": ["tests/test_science.py", "tests/test_data_eval.py"],
            "integration": ["tests/test_engine.py", "tests/test_workflows.py"],
            "notebook": ["tests/test_notebook.py"],
        }[stage]
        coverage = (
            [
                "-p",
                "pytest_cov",
                "--cov=gameformer.geometry",
                "--cov=gameformer.data",
                "--cov=gameformer.model",
                "--cov=gameformer.losses",
                "--cov-fail-under=90",
                "--cov-report=term-missing",
                f"--cov-report=json:{logs / 'scientific-coverage.json'}",
            ]
            if stage == "unit"
            else []
        )
        commands.append(([py, "-m", "pytest", "-q", *coverage, *tests], ROOT, env))
    elif stage == "docs-config":
        commands.append(([py, str(ROOT / "scripts/check_docs.py")], ROOT, env))
    elif stage == "dependencies":
        commands.append(
            (
                [
                    py,
                    str(ROOT / "scripts/audit_dependencies.py"),
                    "--output",
                    str(logs / "dependency-audit.json"),
                ],
                ROOT,
                env,
            )
        )
    with tempfile.TemporaryDirectory(prefix="gameformer-package-") as directory:
        temporary = Path(directory)
        if stage == "package":
            wheel_dir = temporary / "wheels"
            wheel_dir.mkdir()
            commands.extend(
                [
                    ([py, "-m", "build", "--no-isolation", "--outdir", str(wheel_dir)], ROOT, env),
                    (
                        [
                            "uv",
                            "sync",
                            "--frozen",
                            "--no-dev",
                            "--no-install-project",
                            "--python",
                            py,
                        ],
                        ROOT,
                        {**env, "UV_PROJECT_ENVIRONMENT": str(temporary / "clean")},
                    ),
                ]
            )
        started = time.perf_counter()
        with (logs / f"{stage}.log").open("w") as stream:

            def execute(command, cwd, variables):
                stream.write("$ " + " ".join(command) + "\n")
                stream.flush()
                remaining = max(1, timeout - (time.perf_counter() - started))
                subprocess.run(
                    command,
                    cwd=cwd,
                    env=variables,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=remaining,
                )

            try:
                for command, cwd, variables in commands:
                    execute(command, cwd, variables)
                if stage == "package":
                    clean = temporary / "clean/bin/python"
                    wheel = next(wheel_dir.glob("*.whl"))
                    execute(
                        ["uv", "pip", "install", "--python", str(clean), "--no-deps", str(wheel)],
                        temporary,
                        env,
                    )
                    execute(
                        [str(clean), "-c", "import gameformer; print(gameformer.__file__)"],
                        temporary,
                        env,
                    )
                    output = temporary / "run"
                    cli = str(temporary / "clean/bin/gameformer")
                    execute(
                        [
                            cli,
                            "train",
                            "--config",
                            str(ROOT / "configs/smoke.json"),
                            "--synthetic",
                            "--output",
                            str(output),
                        ],
                        temporary,
                        env,
                    )
                    execute(
                        [
                            cli,
                            "eval",
                            "--config",
                            str(ROOT / "configs/smoke.json"),
                            "--synthetic",
                            "--checkpoint",
                            str(output / "latest.pt"),
                            "--output",
                            str(temporary / "eval.json"),
                        ],
                        temporary,
                        env,
                    )
                    assert (
                        json.loads((temporary / "eval.json").read_text())["data_kind"]
                        == "synthetic_diagnostic"
                    )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print((logs / f"{stage}.log").read_text()[-16000:])
                raise
    print(f"{stage}: passed ({time.perf_counter() - started:.2f}s); {logs / (stage + '.log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if shutil.which("uv") is None:
        parser.error("Install uv 0.11.16 first")
    for stage in STAGES if args.stage == "all" else (args.stage,):
        run(stage, args.timeout)


if __name__ == "__main__":
    main()
