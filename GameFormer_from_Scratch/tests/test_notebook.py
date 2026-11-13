import os
import subprocess
import sys

from gameformer.teaching import verify_assets


def test_notebook_inventory_freshness_and_deterministic_assets(tmp_path):
    report = verify_assets(regenerate=tmp_path)
    assert report["figures"] == 18 and report["definitions"] >= 20


def test_walkthrough_executes_offline_in_fresh_cpu_process():
    result = subprocess.run(
        [sys.executable, "-m", "gameformer.teaching", "execute"],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "output and gradient parity passed" in result.stdout
