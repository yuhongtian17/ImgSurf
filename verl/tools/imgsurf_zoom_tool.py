# Copyright 2026 ImgSurf contributors
# Licensed under the Apache License, Version 2.0
"""Stateful v4-style image refinement tool used by the ImgSurf agent loop."""

import math
from typing import Any, Optional
from uuid import uuid4

from PIL import Image
from qwen_vl_utils import fetch_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


class ImgSurfZoomTool(BaseTool):
    """Expose each recursive refinement as a real agent turn.

    The custom agent loop keeps one instance alive across the initial selection
    and all refinement calls. This makes the inner assistant generations part of
    the same replayable policy trajectory instead of hidden environment calls.
    """

    requires_llm = False

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict[str, Any]] = {}
        self.k = float(config.get("k", config.get("expand_ratio", 0.4)))
        self.iou_threshold = float(config.get("iou_threshold", 0.5))
        self.max_iter = int(config.get("max_iter", 4))
        self.max_level = int(config.get("max_level", 1))
        self.expand_mode = str(config.get("expand_mode", "quarter")).lower()
        self.min_pixels = int(config.get("min_pixels", 4096))
        self.max_pixels = int(config.get("max_pixels", 4_194_304))
        self.resize_factor_qwen3 = int(config.get("resize_factor_qwen3", 32))
        self.resize_factor_qwen2 = int(config.get("resize_factor_qwen2", 28))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        image = create_kwargs.get("image", kwargs.get("image"))
        if image is None:
            raise ValueError("image_zoom_in_tool requires the original image")
        model_family = str(create_kwargs.get("model_family", "qwen3_vl"))
        if model_family not in {"qwen3_vl", "qwen2_5_vl"}:
            raise ValueError(f"unsupported model family: {model_family}")
        # The dataset passes a decoded PIL original. qwen_vl_utils.fetch_image
        # may smart-resize PIL inputs, which would silently change the source
        # coordinate system, so bypass it in the normal training path.
        original = (
            image.copy().convert("RGB")
            if isinstance(image, Image.Image)
            else fetch_image({"image": image}).convert("RGB")
        )
        display_size = tuple(create_kwargs.get("display_size") or original.size)
        self._instances[instance_id] = {
            "image": original,
            "question": str(create_kwargs.get("question", "")),
            "model_family": model_family,
            "initial_display_size": (int(display_size[0]), int(display_size[1])),
            "current": None,
            "active_window": None,
            "active_display_size": None,
            "label": "",
            "iterations": 0,
            "converged": False,
        }
        return instance_id, ToolResponse()

    @staticmethod
    def _ordered_finite_box(box: Any) -> Optional[list[float]]:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            values = [float(value) for value in box]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        if values[0] >= values[2] or values[1] >= values[3]:
            return None
        return values

    def validate_parameters(self, instance_id: str, parameters: dict[str, Any]) -> tuple[bool, str]:
        state = self._instances[instance_id]
        box = self._ordered_finite_box(parameters.get("bbox_2d", parameters.get("bbox")))
        label = parameters.get("label")
        if box is None:
            return False, "bbox_2d must contain four finite ordered numbers"
        if not isinstance(label, str) or not label.strip():
            return False, "label must be a non-empty string"
        if state["current"] is None:
            display_size = state["initial_display_size"]
        else:
            display_size = state["active_display_size"]
        max_x, max_y = (1000.0, 1000.0) if state["model_family"] == "qwen3_vl" else display_size
        if box[0] < 0 or box[1] < 0 or box[2] > max_x or box[3] > max_y:
            return False, f"bbox_2d is outside the current coordinate range [0,{max_x}]x[0,{max_y}]"
        return True, ""

    @staticmethod
    def _clamp(box: list[float], image_size: tuple[int, int]) -> list[float]:
        width, height = image_size
        x1 = max(0.0, min(float(width), box[0]))
        y1 = max(0.0, min(float(height), box[1]))
        x2 = max(0.0, min(float(width), box[2]))
        y2 = max(0.0, min(float(height), box[3]))
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def _coordinates_to_window(
        self,
        box: list[float],
        window: list[float],
        display_size: tuple[int, int],
        model_family: str,
    ) -> list[float]:
        wx1, wy1, wx2, wy2 = window
        if model_family == "qwen3_vl":
            scale_x = scale_y = 1000.0
        else:
            scale_x, scale_y = float(display_size[0]), float(display_size[1])
        return [
            wx1 + box[0] / scale_x * (wx2 - wx1),
            wy1 + box[1] / scale_y * (wy2 - wy1),
            wx1 + box[2] / scale_x * (wx2 - wx1),
            wy1 + box[3] / scale_y * (wy2 - wy1),
        ]

    def _expanded_window(self, box: list[float], image_size: tuple[int, int], k_level: float) -> list[float]:
        width, height = float(image_size[0]), float(image_size[1])
        x1, y1, x2, y2 = self._clamp(box, image_size)
        if self.expand_mode == "bbox":
            result = [x1 * (1 - k_level), y1 * (1 - k_level), width * k_level + x2 * (1 - k_level), height * k_level + y2 * (1 - k_level)]
        else:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if self.expand_mode == "ctr":
                result = [
                    min(x1, cx * (1 - k_level)),
                    min(y1, cy * (1 - k_level)),
                    max(x2, width * k_level + cx * (1 - k_level)),
                    max(y2, height * k_level + cy * (1 - k_level)),
                ]
            else:
                result = [
                    min(x1, cx - k_level * width / 2.0),
                    min(y1, cy - k_level * height / 2.0),
                    max(x2, cx + k_level * width / 2.0),
                    max(y2, cy + k_level * height / 2.0),
                ]
        return self._clamp(result, image_size)

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _union(a: list[float], b: list[float]) -> list[float]:
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

    def _resize_crop(
        self, image: Image.Image, box: list[float], max_pixels: int, model_family: str
    ) -> Image.Image:
        width, height = image.size
        integer_box = (
            max(0, int(math.floor(box[0]))),
            max(0, int(math.floor(box[1]))),
            min(width, int(math.ceil(box[2]))),
            min(height, int(math.ceil(box[3]))),
        )
        crop = image.crop(integer_box)
        factor = self.resize_factor_qwen3 if model_family == "qwen3_vl" else self.resize_factor_qwen2
        new_width = max(factor, round(crop.width / factor) * factor)
        new_height = max(factor, round(crop.height / factor) * factor)
        pixels = max(1, crop.width * crop.height)
        if new_width * new_height > max_pixels:
            beta = math.sqrt(pixels / max_pixels)
            new_width = max(factor, math.floor(crop.width / beta / factor) * factor)
            new_height = max(factor, math.floor(crop.height / beta / factor) * factor)
        elif new_width * new_height < self.min_pixels:
            beta = math.sqrt(self.min_pixels / pixels)
            new_width = max(factor, math.ceil(crop.width * beta / factor) * factor)
            new_height = max(factor, math.ceil(crop.height * beta / factor) * factor)
        if crop.size != (new_width, new_height):
            crop = crop.resize((new_width, new_height), Image.Resampling.BICUBIC)
        return crop

    def _k_values(self) -> list[float]:
        return [self.k ** (level + 1) for _ in range(self.max_iter) for level in range(self.max_level)]

    def _bbox_for_zoom_out(self, box: list[float], state: dict[str, Any]) -> list[int]:
        if state["model_family"] == "qwen3_vl":
            width, height = state["image"].size
            return [
                int(round(box[0] / width * 1000)),
                int(round(box[1] / height * 1000)),
                int(round(box[2] / width * 1000)),
                int(round(box[3] / height * 1000)),
            ]
        return [int(round(value)) for value in box]

    def _refinement_observation(self, state: dict[str, Any]) -> tuple[ToolResponse, dict[str, Any]]:
        k_values = self._k_values()
        k_level = k_values[state["iterations"]]
        window = self._expanded_window(state["current"], state["image"].size, k_level)
        crop = self._resize_crop(
            state["image"],
            window,
            max(1, int(self.max_pixels * k_level**2)),
            state["model_family"],
        )
        state["active_window"] = window
        state["active_display_size"] = crop.size
        index = state["iterations"] + 1
        metadata = {
            "success": True,
            "done": False,
            "converged": False,
            "iteration": state["iterations"],
            "index": index,
            "k_level": k_level,
            "window": window,
            "zoom_out_bbox": self._bbox_for_zoom_out(window, state),
            "display_size": crop.size,
            "label": state["label"],
            "question": state["question"],
            "model_family": state["model_family"],
        }
        return ToolResponse(image=[crop]), metadata

    def _final_observation(
        self, state: dict[str, Any], final_box: list[float]
    ) -> tuple[ToolResponse, dict[str, Any]]:
        crop = self._resize_crop(state["image"], final_box, self.max_pixels, state["model_family"])
        rounded = [int(round(value)) for value in final_box]
        text = (
            f"Final refined crop; original-image bbox={rounded}; iterations={state['iterations']}; "
            f"converged={str(state['converged']).lower()}."
        )
        metadata = {
            "success": True,
            "done": True,
            "converged": state["converged"],
            "iterations": state["iterations"],
            "final_bbox": final_box,
            "display_size": crop.size,
            "label": state["label"],
            "question": state["question"],
            "model_family": state["model_family"],
        }
        return ToolResponse(text=text, image=[crop]), metadata

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instances[instance_id]
        valid, error = self.validate_parameters(instance_id, parameters)
        if not valid:
            return ToolResponse(text=f"Error: {error}."), -0.10, {
                "success": False,
                "parsed_valid": False,
                "done": False,
            }
        box = self._ordered_finite_box(parameters.get("bbox_2d", parameters.get("bbox")))
        assert box is not None
        state["label"] = str(parameters["label"]).strip()
        if state["current"] is None:
            width, height = state["image"].size
            state["current"] = self._coordinates_to_window(
                box,
                [0.0, 0.0, float(width), float(height)],
                state["initial_display_size"],
                state["model_family"],
            )
            response, metadata = self._refinement_observation(state)
            metadata["parsed_valid"] = True
            return response, 0.0, metadata

        refined = self._coordinates_to_window(
            box,
            state["active_window"],
            state["active_display_size"],
            state["model_family"],
        )
        previous = state["current"]
        state["iterations"] += 1
        state["converged"] = self._iou(previous, refined) > self.iou_threshold
        final_box = self._union(previous, refined)
        if state["converged"] or state["iterations"] >= len(self._k_values()):
            response, metadata = self._final_observation(state, final_box)
        else:
            state["current"] = refined
            response, metadata = self._refinement_observation(state)
        metadata["parsed_valid"] = True
        return response, 0.0, metadata

    async def fallback_after_invalid_refinement(
        self, instance_id: str
    ) -> tuple[ToolResponse, float, dict]:
        """Match v4's missing-bbox fallback by selecting the whole shown crop."""

        state = self._instances[instance_id]
        if state["current"] is None or state["active_display_size"] is None:
            return ToolResponse(text="Error: no active refinement window."), -0.10, {
                "success": False,
                "parsed_valid": False,
                "done": True,
            }
        width, height = state["active_display_size"]
        if state["model_family"] == "qwen3_vl":
            box = [0.0, 0.0, 1000.0, 1000.0]
        else:
            box = [0.0, 0.0, float(width), float(height)]
        parameters = {"bbox_2d": box, "label": state["label"] or "relevant evidence"}
        response, reward, metadata = await self.execute(instance_id, parameters)
        metadata["parsed_valid"] = False
        metadata["fallback"] = True
        return response, reward, metadata

    async def finalize_current(self, instance_id: str) -> tuple[ToolResponse, float, dict]:
        """Stop refinement early when the remaining token budget is too small."""

        state = self._instances[instance_id]
        if state["current"] is None:
            return ToolResponse(text="No valid crop is available."), 0.0, {
                "success": False,
                "done": True,
                "converged": False,
                "iterations": state["iterations"],
            }
        response, metadata = self._final_observation(state, state["current"])
        metadata["context_limited"] = True
        return response, 0.0, metadata

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
