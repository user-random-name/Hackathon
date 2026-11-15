from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

SOURCE_CLASS_ID = 671
SOURCE_INTERNAL_LABEL = "mountain bike"
TARGET_CLASS_ID = 665
TARGET_LABEL = "moped"
MIN_TARGET_CONFIDENCE = 0.99
MAX_PIXEL_DELTA = 1
MAX_OUTPUT_BYTES = 1024 * 1024
IMAGE_FORMAT = "PNG"
IMAGE_MODE = "RGB"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = MobileNet_V3_Large_Weights.IMAGENET1K_V2.meta["categories"]


@dataclass(frozen=True)
class Prediction:
    """Top-1 label and softmax probabilities for one image."""

    class_id: int
    class_name: str
    confidence: float
    target_confidence: float


def load_model(
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load the supplied MobileNetV3 Large checkpoint in evaluation mode."""
    model = mobilenet_v3_large(weights=None)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def pil_to_tensor(image: Image.Image, *, device: str | torch.device = "cpu") -> Tensor:
    """Convert an RGB image to a (1, 3, H, W) tensor without resizing."""
    if image.mode != IMAGE_MODE:
        raise ValueError(f"Expected RGB image, received mode {image.mode!r}.")

    array = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32).div(255.0)


def tensor_to_rgb_image(tensor: Tensor) -> Image.Image:
    """Clamp and round one CHW or 1xCHW tensor to RGB integers for PNG saving."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Only a single image can be saved.")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3 or min(tensor.shape[1:]) <= 0:
        raise ValueError(f"Unexpected tensor shape {tuple(tensor.shape)}.")

    pixels = tensor.detach().cpu().clamp(0.0, 1.0).mul(255.0).round()
    array = pixels.to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def normalize_batch(batch: Tensor) -> Tensor:
    """Apply ImageNet normalization to NCHW tensors, preserving gradients."""
    mean = batch.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = batch.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (batch - mean) / std


@torch.inference_mode()
def predict_tensor(model: nn.Module, batch: Tensor) -> Prediction:
    if batch.ndim != 4 or tuple(batch.shape[:2]) != (1, 3) or min(batch.shape[2:]) <= 0:
        raise ValueError(
            f"Expected one RGB tensor with shape (1, 3, H, W), received {tuple(batch.shape)}."
        )

    probabilities = model(normalize_batch(batch)).softmax(dim=1)[0]
    class_id = probabilities.argmax().item()
    return Prediction(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        confidence=probabilities[class_id].item(),
        target_confidence=probabilities[TARGET_CLASS_ID].item(),
    )
