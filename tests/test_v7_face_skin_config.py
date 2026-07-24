from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def test_v7_configs_only_relax_the_face_skin_density_gate() -> None:
    expected_experiment = (
        "custom_one_to_one_reference_color_v7_face_skin"
    )
    for flavor in ("server", "example"):
        v6_path = (
            CONFIG_ROOT
            / f"custom_unpaired_reference_v6_face_skin.{flavor}.yaml"
        )
        v7_path = (
            CONFIG_ROOT
            / f"custom_unpaired_reference_v7_face_skin.{flavor}.yaml"
        )
        v6 = yaml.safe_load(v6_path.read_text(encoding="utf-8"))
        v7 = yaml.safe_load(v7_path.read_text(encoding="utf-8"))

        expected = deepcopy(v6)
        expected["experiment"] = expected_experiment
        expected["pipeline"]["params"][
            "reference_skin_face_density_max"
        ] = 1.0

        assert v7 == expected
        assert v7["save_dir"] == "../experiment"
        assert v7["resume"] is False


def test_v7_launcher_targets_v7_server_config() -> None:
    v6_launcher = (
        PROJECT_ROOT
        / "scripts/train_custom_unpaired_reference_v6_face_skin.sh"
    ).read_text(encoding="utf-8")
    v7_path = (
        PROJECT_ROOT
        / "scripts/train_custom_unpaired_reference_v7_face_skin.sh"
    )

    assert v7_path.read_text(encoding="utf-8") == v6_launcher.replace(
        "v6_face_skin",
        "v7_face_skin",
    )
    assert v7_path.stat().st_mode & 0o111
