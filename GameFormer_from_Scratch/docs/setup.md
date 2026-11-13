# Setup and environment

The tested local platform is Linux x86-64, Python 3.11.14, torch 2.15.0+cpu,
NumPy 2.4.6 and uv 0.11.16. Exact transitive requirements are in `uv.lock`.
GitHub CI declares Python 3.11 and 3.12 CPU jobs; local success on 3.11 alone
does not establish 3.12 or hosted CI success. Python >=3.11 is required.
System Python 3.10 / Torch 1.8 is unsuitable for these entry points.

```bash
python3.11 -m pip install uv==0.11.16
uv sync --frozen --extra dev --extra tracking
.venv/bin/gameformer --help
```

For the inspected host, Python was supplied explicitly:

```bash
UV_CACHE_DIR=/tmp/gameformer-uv-cache uv sync --frozen --extra dev --extra tracking \
  --python /home/beomseokkim2/miniconda3/envs/torch_env/bin/python
```

The base dependencies are PyTorch, NumPy and Matplotlib. Standard PyTorch layers
implement the learnable model; NumPy handles NPZ caches; Matplotlib renders
diagnostic trajectories. Optional MLflow provides tracking, not a training loop.
Development dependencies provide tests, lint/types, notebooks, deterministic
figure rendering, packaging and vulnerability auditing. SVG and PNG share layout
primitives; Pillow rasterizes the PNG with Matplotlib's wheel-bundled DejaVu fonts.
The manifest records renderer versions and font hashes. Host Cairo and Fontconfig
are not part of figure generation.
The pinned setuptools backend supports building without a separate wheel package.

The lock explicitly selects the PyTorch CPU wheel index. CUDA and TorchVision
are not used in the tested runtime. A CUDA experiment needs an isolated compatible
PyTorch/CUDA environment, its own dependency record, and numerical/memory checks.
No GPU claim follows from this CPU lock. TensorFlow/Waymo is intentionally outside
the base environment and must be validated as an optional SDK boundary.

For deterministic and quick CPU checks use `OMP_NUM_THREADS=1`. In the restricted
workspace also use `UV_CACHE_DIR=/tmp/gameformer-uv-cache` and
`MPLCONFIGDIR=/tmp/gameformer-mpl`. The test launcher disables unrelated global
pytest plugins. No dataset, environment, checkpoint or MLflow database is committed.

Run `python scripts/check_all.py` with the environment interpreter. It retains
logs under `outputs/checks/`, builds a distribution, creates a clean locked base
environment, and exercises installed train/eval commands from another directory.
Dependency audit results reflect the database at audit time, not a permanent
absence of vulnerabilities. The [audit evidence](dependency-audit.json) records
the query protocol: the CPU wheel's `+cpu` suffix is mapped to its upstream
public release for advisory lookup. Installation still uses the exact hashed
CPU wheel in `uv.lock`. Skipped queries and known advisories fail the gate;
no package or CVE is silently ignored.

Mypy checks stable config, geometry/data, tracking and trainer interfaces.
Model tensor axes are enforced by analytical/runtime fixtures rather than static
shape typing. Preparation, optional SDK glue, performance experiments and teaching
generation are linted and behaviorally checked but are not yet in the typed set.
