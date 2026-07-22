import torch

from cm_kan.ml.pipelines.unsupervised import UnsupervisedPipeline


def test_chroma_loss_allows_multiplicative_brightness_change() -> None:
    image = torch.tensor([0.62, 0.41, 0.28]).view(1, 3, 1, 1)
    image = image.expand(1, 3, 32, 32)

    loss = UnsupervisedPipeline._chroma_loss(image * 0.45, image)

    assert loss.item() < 1e-6


def test_chroma_loss_detects_hue_shift() -> None:
    image = torch.tensor([0.62, 0.41, 0.28]).view(1, 3, 1, 1)
    image = image.expand(1, 3, 32, 32)
    shifted = image.clone()
    shifted[:, 0] = shifted[:, 0] * 0.55

    loss = UnsupervisedPipeline._chroma_loss(shifted, image)

    assert loss.item() > 0.05


def test_reflectance_loss_allows_global_brightness_change() -> None:
    generator = torch.Generator().manual_seed(7)
    image = torch.rand((1, 3, 32, 32), generator=generator) * 0.7 + 0.2

    loss = UnsupervisedPipeline._reflectance_loss(image * 0.5, image)

    assert loss.item() < 1e-5


def test_reflectance_loss_detects_local_tone_change() -> None:
    image = torch.full((1, 3, 32, 32), 0.6)
    changed = image.clone()
    changed[:, :, 8:24, 8:24] *= 0.45

    loss = UnsupervisedPipeline._reflectance_loss(changed, image)

    assert loss.item() > 0.05


def test_patch_nce_prefers_matching_spatial_features() -> None:
    key = torch.eye(4).reshape(1, 4, 2, 2)
    matching_query = key.clone().requires_grad_(True)
    shifted_query = torch.roll(key, shifts=1, dims=-1)

    matching_loss = UnsupervisedPipeline._patch_nce_loss(
        [matching_query], [key], num_patches=4, temperature=0.07
    )
    shifted_loss = UnsupervisedPipeline._patch_nce_loss(
        [shifted_query], [key], num_patches=4, temperature=0.07
    )

    assert matching_loss.item() < shifted_loss.item()
    matching_loss.backward()
    assert matching_query.grad is not None
    assert torch.isfinite(matching_query.grad).all()
