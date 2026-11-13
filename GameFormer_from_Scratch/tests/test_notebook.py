import os
import subprocess
import sys

from gameformer.teaching import verify_assets


def test_figure_rendering_does_not_depend_on_host_fontconfig(tmp_path):
    from pathlib import Path

    fontconfig = tmp_path / "fonts.conf"
    fontconfig.write_text(f"<fontconfig><dir>{tmp_path / 'absent'}</dir></fontconfig>")
    script = (
        "import sys; from gameformer.teaching import draw_figure, FIGURES; "
        "from pathlib import Path; path=Path(sys.argv[1]); path.mkdir(); "
        "draw_figure(FIGURES[0], path)"
    )
    for name, extra in (("normal", {}), ("empty-fontconfig", {"FONTCONFIG_FILE": str(fontconfig)})):
        subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / name)],
            check=True,
            env={**os.environ, **extra},
        )
    assert (
        Path(tmp_path / "normal/overview.png").read_bytes()
        == Path(tmp_path / "empty-fontconfig/overview.png").read_bytes()
    )


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
