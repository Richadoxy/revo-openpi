"""Policy transforms for Revo bimanual Revo3 datasets.

The current Revo LeRobot v2.1 datasets store a 70D state/action vector:

- 0:14   left/right end-effector poses, 7D each
- 14:28  left/right arm joints, 7D each
- 28:70  left/right Revo3 hand joints, 21D each

For pi05 fine-tuning we train on the 56D joint-only slice, dropping the first
14 pose dimensions and keeping:

- 0:7    left arm joints
- 7:14   right arm joints
- 14:35  left Revo3 hand joints
- 35:56  right Revo3 hand joints
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


REVO_RAW_DIM = 70
REVO_ACTION_DIM = 56
REVO_JOINT_SLICE = slice(14, 70)


def make_revo_example() -> dict:
    """Creates a random input example for the Revo policy."""
    return {
        "observation/state": np.random.rand(REVO_RAW_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick and place",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _joint_slice(value: np.ndarray, *, key: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] == REVO_ACTION_DIM:
        return arr
    if arr.shape[-1] != REVO_RAW_DIM:
        raise ValueError(f"{key} must have trailing dim {REVO_RAW_DIM} or {REVO_ACTION_DIM}, got {arr.shape}")
    return arr[..., REVO_JOINT_SLICE]


@dataclasses.dataclass(frozen=True)
class RevoInputs(transforms.DataTransformFn):
    """Converts Revo dataset/inference observations into the pi0/pi05 model format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": _joint_slice(data["observation/state"], key="observation/state"),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _joint_slice(data["actions"], key="actions")

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RevoOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the 56D Revo joint action format."""

    action_dim: int = REVO_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[..., : self.action_dim]}
