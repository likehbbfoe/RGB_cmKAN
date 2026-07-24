from copy import deepcopy
from pathlib import Path

import yaml

from cm_kan.core.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
ENVIRONMENT_PARAMETER_NAMES = (
    "reference_environment_weight",
    "reference_environment_ramp_epochs",
    "reference_environment_chroma_std_weight",
    "reference_environment_luminance_weight",
    "reference_environment_luminance_std_weight",
    "reference_environment_face_dilation",
    "reference_environment_min_cell_fraction",
)


def test_v8_configs_add_only_the_environment_training_recipe() -> None:
    expected_experiment = (
        "custom_one_to_one_reference_color_v8_environment"
    )
    environment_params = {
        "reference_environment_weight": 2.0,
        "reference_environment_ramp_epochs": 10,
        "reference_environment_chroma_std_weight": 0.20,
        "reference_environment_luminance_weight": 0.25,
        "reference_environment_luminance_std_weight": 0.10,
        "reference_environment_face_dilation": 9,
        "reference_environment_min_cell_fraction": 0.05,
    }
    for flavor in ("server", "example"):
        v7_path = (
            CONFIG_ROOT
            / f"custom_unpaired_reference_v7_face_skin.{flavor}.yaml"
        )
        v8_path = (
            CONFIG_ROOT
            / f"custom_unpaired_reference_v8_environment.{flavor}.yaml"
        )
        v7 = yaml.safe_load(v7_path.read_text(encoding="utf-8"))
        v8 = yaml.safe_load(v8_path.read_text(encoding="utf-8"))

        expected = deepcopy(v7)
        expected["experiment"] = expected_experiment
        expected_params = expected["pipeline"]["params"]
        expected_params["exposure_weight"] = 0.5
        expected_params["reference_style_weight"] = 5.0
        insertion_point = "reference_skin_tone_weight"
        reordered_params = {}
        for name, value in expected_params.items():
            if name == insertion_point:
                reordered_params.update(environment_params)
            reordered_params[name] = value
        expected["pipeline"]["params"] = reordered_params

        assert v8 == expected
        assert v8["data"]["params"]["crop_size"] == 256
        assert v8["pipeline"]["params"]["batch_size"] == 8
        assert v8["save_dir"] == "../experiment"
        assert v8["resume"] is False


def test_v8_config_schema_preserves_every_environment_parameter() -> None:
    path = (
        CONFIG_ROOT
        / "custom_unpaired_reference_v8_environment.server.yaml"
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    config = Config(**raw)
    params = config.pipeline.params

    assert params.reference_environment_weight == 2.0
    assert params.reference_environment_ramp_epochs == 10
    assert params.reference_environment_chroma_std_weight == 0.20
    assert params.reference_environment_luminance_weight == 0.25
    assert params.reference_environment_luminance_std_weight == 0.10
    assert params.reference_environment_face_dilation == 9
    assert params.reference_environment_min_cell_fraction == 0.05


def test_selector_forwards_every_environment_parameter() -> None:
    source = (
        PROJECT_ROOT / "cm_kan/core/selector/pipeline.py"
    ).read_text(encoding="utf-8")

    for parameter_name in ENVIRONMENT_PARAMETER_NAMES:
        assert f"{parameter_name}=(" in source
        assert f"config.pipeline.params.{parameter_name}" in source


def test_v8_launcher_targets_v8_server_config() -> None:
    v7_launcher = (
        PROJECT_ROOT
        / "scripts/train_custom_unpaired_reference_v7_face_skin.sh"
    ).read_text(encoding="utf-8")
    v8_path = (
        PROJECT_ROOT
        / "scripts/train_custom_unpaired_reference_v8_environment.sh"
    )

    expected = v7_launcher.replace(
        "v7_face_skin",
        "v8_environment",
    )
    assert v8_path.read_text(encoding="utf-8") == expected
    assert v8_path.stat().st_mode & 0o111
