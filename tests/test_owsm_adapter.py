from pathlib import Path

from models import create_audio_model, parameter_count


def test_owsm_base_loads_from_project_cache_without_network():
    root = Path(__file__).resolve().parents[1]

    model = create_audio_model("owsm_v3.1_base", root)

    assert 100_000_000 <= parameter_count(model) <= 102_000_000
