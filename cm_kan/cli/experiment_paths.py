from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(path: str) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def experiment_directory(save_dir: str, experiment: str) -> str:
    """Return one normalized experiment directory for all CLI commands."""
    return str((_project_path(save_dir) / experiment).resolve())


def prediction_output_directory(
    save_dir: str,
    experiment: str,
    output: str | None,
) -> str:
    """Resolve an explicit output or the experiment-local default."""
    if output is not None:
        return str(_project_path(output))
    return str(
        Path(
            experiment_directory(save_dir, experiment)
        ) / "predictions"
    )
