import os
import subprocess
import sys
from pathlib import Path


def test_gn_plotting_forces_a_noninteractive_backend_before_matplotlib_import():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("MPLBACKEND", None)
    environment["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import gn_experiment, matplotlib; print(matplotlib.get_backend())",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip().lower() == "agg"
