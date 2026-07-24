import os
from typing import Any


def domain_path(data_root: str, split: str, domain: str) -> str:
    """Resolve one domain directory in a split-based custom dataset."""
    split_root = os.path.join(data_root, split)
    path = (
        os.path.join(split_root, domain)
        if os.path.isdir(split_root)
        else os.path.join(data_root, domain)
    )
    real_path = os.path.join(path, "real")
    if split == "train" and os.path.isdir(real_path):
        return real_path
    return path


def override_data_root(
    config: dict[str, Any],
    data_root: str,
    source_domain: str,
    target_domain: str,
) -> None:
    """Apply a custom dataset root to a raw configuration dictionary."""
    if config.get("data", {}).get("type") != "custom_unpaired":
        raise ValueError("--data-root can only be used with data.type=custom_unpaired")

    data_config = config["data"]
    data_config["train"] = {
        "source": domain_path(data_root, "train", source_domain),
        "target": domain_path(data_root, "train", target_domain),
    }

    for split in ("val", "test"):
        if os.path.isdir(os.path.join(data_root, split)):
            data_config[split] = {
                "source": domain_path(data_root, split, source_domain),
                "target": domain_path(data_root, split, target_domain),
            }
        else:
            data_config.pop(split, None)


def override_face_mask_root(
    config: dict[str, Any],
    face_mask_root: str,
) -> None:
    """Apply a mirrored face-mask sidecar root to a raw configuration."""
    if config.get("data", {}).get("type") != "custom_unpaired":
        raise ValueError(
            "--face-mask-root can only be used with "
            "data.type=custom_unpaired"
        )
    config["data"].setdefault("params", {})["face_mask_root"] = (
        face_mask_root
    )
