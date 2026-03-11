import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _normalize(x: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    denom = (max_val - min_val)
    denom = np.where(denom == 0, 1.0, denom)
    return (x - min_val) / denom


def _unnormalize(x: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    return x * (max_val - min_val) + min_val


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _resize_uint8_hwc(img_hwc_uint8: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """
    Resize helper. Replace with cv2 / project-native util if desired.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Install pillow or replace _resize_uint8_hwc().") from e

    out_h, out_w = out_hw
    pil = Image.fromarray(img_hwc_uint8)
    pil = pil.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _decode_rgb_hwc_float01_to_uint8(img: np.ndarray, *, out_hw: tuple[int, int]) -> np.ndarray:
    """
    Input: HWC float in [0,1] (your dataset stats show min/max are 0..1), or uint8 HWC.
    Output: uint8 HWC resized to out_hw.
    """
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))

    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        img = (255.0 * img).astype(np.uint8)

    if (img.shape[0], img.shape[1]) != out_hw:
        img = _resize_uint8_hwc(img, out_hw)

    return img


# ---- Limits (your robot) ----

def _joint_limits() -> np.ndarray:
    # shape (7,2): [[min,max], ...]
    return np.array(
        [
            [-2.96705972839036, 2.96705972839036],
            [-2.0943951023931953, 2.0943951023931953],
            [-2.96705972839036, 2.96705972839036],
            [-2.0943951023931953, 2.0943951023931953],
            [-2.96705972839036, 2.96705972839036],
            [-2.0943951023931953, 2.0943951023931953],
            [-2.96705972839036, 2.96705972839036],
        ],
        dtype=np.float32,
    )


GRIPPER_MIN = np.float32(-108.0)
GRIPPER_MAX = np.float32(130.0)


def _state_action_minmax_8d() -> tuple[np.ndarray, np.ndarray]:
    jl = _joint_limits()
    joint_min = jl[:, 0]
    joint_max = jl[:, 1]
    mn = np.concatenate([joint_min, np.array([GRIPPER_MIN], dtype=np.float32)], axis=0)
    mx = np.concatenate([joint_max, np.array([GRIPPER_MAX], dtype=np.float32)], axis=0)
    return mn, mx


def _encode_8d_to_unit(x: np.ndarray) -> np.ndarray:
    """
    Normalize 7 joints + 1 gripper to [0,1] using joint limits + gripper limits.
    Works for shape (8,) or (T,8) (or any (...,8)).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != 8:
        raise ValueError(f"Expected last dim=8 (7 joints + gripper), got {x.shape}")

    mn, mx = _state_action_minmax_8d()
    y = _normalize(x, mn, mx)

    # Pad action space to 32, according to PI05

    return _clip01(y)


def _decode_8d_from_unit(y: np.ndarray) -> np.ndarray:
    """
    Inverse of _encode_8d_to_unit for inference-time action unnormalization.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.shape[-1] != 8:
        raise ValueError(f"Expected last dim=8 (7 joints + gripper), got {y.shape}")

    mn, mx = _state_action_minmax_8d()
    return _unnormalize(y, mn, mx)


@dataclasses.dataclass(frozen=True)
class LeRobotLWRInputs(transforms.DataTransformFn):
    """
    Inputs transform for your LeRobot v3 dataset.
    Produces the same structure as aloha_policy.py uses for PI models.
    """

    model_type: _model.ModelType

    # Downscale 720x1280 -> 224x224 by default
    out_hw: tuple[int, int] = (224, 224)

    # If True, normalize state/actions using joint/gripper limits.
    adapt_to_pi: bool = True

    STATIC_KEY: ClassVar[str] = "observation.static_cam.rgb"
    WRIST_KEY: ClassVar[str] = "observation.wrist_cam.rgb"
    STATE_KEY: ClassVar[str] = "state"
    ACTION_KEY: ClassVar[str] = "actions"

    def __call__(self, data: dict) -> dict:
        if self.STATE_KEY not in data:
            raise ValueError(f"Missing key: {self.STATE_KEY}")
        if self.STATIC_KEY not in data:
            raise ValueError(f"Missing key: {self.STATIC_KEY}")

        # --- images ---
        base_image = _decode_rgb_hwc_float01_to_uint8(data[self.STATIC_KEY], out_hw=self.out_hw)

        if self.WRIST_KEY in data:
            wrist_image = _decode_rgb_hwc_float01_to_uint8(data[self.WRIST_KEY], out_hw=self.out_hw)
            left_wrist_mask = np.True_
        else:
            wrist_image = np.zeros_like(base_image)
            left_wrist_mask = np.False_

        # Pad missing right wrist cam
        right_wrist_image = np.zeros_like(base_image)

        # --- state ---
        state = np.asarray(data[self.STATE_KEY], dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"Expected {self.STATE_KEY} shape (8,), got {state.shape}")

        if self.adapt_to_pi:
            state = _encode_8d_to_unit(state)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_wrist_mask,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # --- actions (training only) ---
        if self.ACTION_KEY in data:
            actions = np.asarray(data[self.ACTION_KEY], dtype=np.float32)

            # allow (8,) or (T,8)
            if actions.shape == (8,):
                if self.adapt_to_pi:
                    actions = _encode_8d_to_unit(actions)
            elif actions.ndim == 2 and actions.shape[1] == 8:
                if self.adapt_to_pi:
                    actions = _encode_8d_to_unit(actions)
            else:
                raise ValueError(f"Expected action shape (8,) or (T,8), got {actions.shape}")

            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LeRobotLWROutputs(transforms.DataTransformFn):
    """
    Output transform: model -> robot command space.
    """

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            raise ValueError("Model output missing key: actions")

        actions = np.asarray(data["actions"], dtype=np.float32)

        if self.adapt_to_pi:
            actions = _decode_8d_from_unit(actions)

        return {"actions": actions}
