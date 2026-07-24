import ast
from pathlib import Path

import yaml

from cm_kan.cli.experiment_paths import (
    experiment_directory,
    prediction_output_directory,
)
from scripts.preview_skin_masks import DEFAULT_OUTPUT
from scripts.report_reference_metrics import FALLBACK_METRICS_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
EXTERNAL_SAVE_DIR = "../experiment"


def test_all_shipped_configs_save_experiments_outside_repository() -> None:
    config_paths = sorted(CONFIG_ROOT.glob("*.yaml"))

    assert config_paths
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["save_dir"] == EXTERNAL_SAVE_DIR, config_path


def test_core_config_default_uses_external_experiment_directory() -> None:
    config_source = (
        PROJECT_ROOT / "cm_kan/core/config/config.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(config_source)
    config_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Config"
    )
    save_dir_assignment = next(
        node
        for node in config_class.body
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "save_dir"
        )
    )

    assert ast.literal_eval(save_dir_assignment.value) == EXTERNAL_SAVE_DIR


def test_pretrained_checkpoints_do_not_point_to_legacy_directory() -> None:
    for config_path in CONFIG_ROOT.glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pretrained_model = (
            config.get("pipeline", {})
            .get("params", {})
            .get("pretrained_model")
        )
        if pretrained_model:
            assert pretrained_model.startswith(f"{EXTERNAL_SAVE_DIR}/")


def test_auxiliary_report_defaults_use_external_experiment_directory() -> None:
    external_root = (PROJECT_ROOT.parent / "experiment").resolve()

    assert FALLBACK_METRICS_PATH.is_relative_to(external_root)
    assert (
        FALLBACK_METRICS_PATH
        == external_root
        / "custom_one_to_one_reference_color_v7_face_skin"
        / "logs/metrics.csv"
    )
    assert DEFAULT_OUTPUT.is_relative_to(external_root)


def test_cli_resolves_experiment_directory_from_project_root() -> None:
    expected = (
        PROJECT_ROOT.parent / "experiment" / "example_run"
    ).resolve()

    assert Path(
        experiment_directory(EXTERNAL_SAVE_DIR, "example_run")
    ) == expected
    assert Path(
        prediction_output_directory(
            EXTERNAL_SAVE_DIR,
            "example_run",
            output=None,
        )
    ) == expected / "predictions"


def test_explicit_relative_prediction_output_uses_project_root() -> None:
    assert Path(
        prediction_output_directory(
            EXTERNAL_SAVE_DIR,
            "example_run",
            output="../experiment/results/custom",
        )
    ) == (
        PROJECT_ROOT.parent / "experiment/results/custom"
    ).resolve()
