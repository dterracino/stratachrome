"""
segmentation.py
---------------
Isolates foreground subjects from backgrounds using BiRefNet.
Produces soft alpha mattes with configurable boundary feathering and
partitions perceptual lightness arrays into distinct two-tier channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import AutoModelForImageSegmentation

_DEFAULT_MODEL_ID = "ZhengPeng7/BiRefNet"
_MODEL_INPUT_DIMENSION = 1024


@dataclass(frozen=True)
class SegmentationConfig:
    """Runtime configuration for the BiRefNet foreground extraction model.

    Attributes:
        model_id: Hugging Face model identifier.
        device: Computation target ('auto', 'cuda', or 'cpu').
        feather_radius: Gaussian blur kernel radius for edge softness.
        threshold: Binarization cut-off; None preserves soft alpha gradients.
    """

    model_id: str = _DEFAULT_MODEL_ID
    device: Optional[str] = "auto"
    feather_radius: int = 2
    threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.device not in {None, "auto", "cuda", "cpu"}:
            raise ValueError(f"device must be 'auto', 'cuda', or 'cpu', got {self.device!r}")
        if self.feather_radius < 0:
            raise ValueError(f"feather_radius must be non-negative, got {self.feather_radius}")
        if self.threshold is not None and not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {self.threshold}")


@dataclass(frozen=True)
class SegmentationResult:
    """Represents the output of the foreground segmentation step.

    Attributes:
        matte: 2D float32 array in [0.0, 1.0] matching source image dimensions.
               1.0 represents definite foreground, 0.0 represents background.
    """

    matte: np.ndarray

    def __post_init__(self) -> None:
        if self.matte.ndim != 2:
            raise ValueError(f"Matte must be 2D, got ndim={self.matte.ndim}")


def _resolve_compute_device(requested: Optional[str]) -> torch.device:
    """Determines the target torch device based on availability and request."""
    if requested not in {None, "auto"}:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_birefnet_model(model_id: str, device: torch.device) -> torch.nn.Module:
    """Loads and initializes BiRefNet weights on the specified device."""
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if device.type == "cuda":
        model = model.half()
    model.to(device)
    model.eval()
    return model


def _build_input_transform() -> T.Compose:
    """Constructs the standard image preprocessing transform for BiRefNet."""
    return T.Compose(
        [
            T.Resize((_MODEL_INPUT_DIMENSION, _MODEL_INPUT_DIMENSION)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _run_inference(
    model: torch.nn.Module,
    pil_image: Image.Image,
    device: torch.device,
) -> np.ndarray:
    """Runs BiRefNet forward pass and rescales sigmoid mask to original size."""
    transform = _build_input_transform()
    transformed_tensor = cast(torch.Tensor, transform(pil_image))
    input_tensor = transformed_tensor.unsqueeze(0).to(device)

    # Match input tensor dtype with model weights (e.g. half/float16 on CUDA)
    first_param = next(model.parameters(), None)
    if first_param is not None and input_tensor.dtype != first_param.dtype:
        input_tensor = input_tensor.to(dtype=first_param.dtype)

    with torch.no_grad():
        preds = model(input_tensor)
        # BiRefNet outputs a list/tuple of intermediate masks; index -1 is the final prediction
        if isinstance(preds, (list, tuple)):
            raw_pred = preds[-1]
        else:
            raw_pred = preds
        sigmoid_mask = torch.sigmoid(raw_pred).squeeze().float().cpu().numpy()

    # Rescale to native image dimensions (PIL uses width, height; cv2 uses cols, rows)
    w, h = pil_image.size
    rescaled_matte = cv2.resize(sigmoid_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(rescaled_matte, 0.0, 1.0).astype(np.float32)


def _apply_feathering(matte: np.ndarray, radius: int) -> np.ndarray:
    """Applies a normalized Gaussian blur kernel to soften edge transitions."""
    if radius == 0:
        return matte
    kernel_size = 2 * radius + 1
    blurred = cv2.GaussianBlur(matte, (kernel_size, kernel_size), sigmaX=radius / 2.0)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def _apply_threshold(matte: np.ndarray, threshold: float) -> np.ndarray:
    """Binarizes the continuous matte based on a user-specified cutoff."""
    return (matte >= threshold).astype(np.float32)


def partition_lightness_channels(
    full_lightness: np.ndarray,
    matte: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Partitions perceptual lightness into background and foreground channels.

    Args:
        full_lightness: 2D array of L* values [0.0, 100.0] covering the entire image.
        matte: 2D alpha matte array [0.0, 1.0] matching full_lightness dimensions.

    Returns:
        tuple (bg_lightness, fg_lightness):
            - bg_lightness: Full lightness matrix for background tier mapping.
            - fg_lightness: Full lightness matrix for foreground tier mapping.
    """
    if full_lightness.shape != matte.shape:
        raise ValueError(f"Shape mismatch: lightness {full_lightness.shape} vs matte {matte.shape}")

    bg_lightness = full_lightness.copy().astype(np.float32)
    fg_lightness = full_lightness.copy().astype(np.float32)
    return bg_lightness, fg_lightness


class ForegroundSegmenter:
    """Stateful worker encapsulating the BiRefNet model and matte pipeline."""

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        self._config = config or SegmentationConfig()
        self._device = _resolve_compute_device(self._config.device)
        self._model: Optional[torch.nn.Module] = None

    @property
    def config(self) -> SegmentationConfig:
        return self._config

    @property
    def device(self) -> torch.device:
        return self._device

    def _ensure_model_loaded(self) -> torch.nn.Module:
        """Lazy-loads BiRefNet weights upon first call."""
        if self._model is None:
            self._model = _load_birefnet_model(self._config.model_id, self._device)
        return self._model

    def extract_matte(self, image: Image.Image | np.ndarray) -> SegmentationResult:
        """Runs segmentation on an RGB image and returns the post-processed matte.

        Args:
            image: PIL Image or NumPy array in RGB uint8 format.

        Returns:
            SegmentationResult holding the feathered 2D alpha matte in [0.0, 1.0].
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        model = self._ensure_model_loaded()
        raw_matte = _run_inference(model, pil_img, self._device)

        processed = raw_matte
        if self._config.threshold is not None:
            processed = _apply_threshold(processed, self._config.threshold)

        if self._config.feather_radius > 0:
            processed = _apply_feathering(processed, self._config.feather_radius)

        return SegmentationResult(matte=processed)
