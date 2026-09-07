# Copyright 2026 ImgSurf contributors
# Licensed under the Apache License, Version 2.0
"""Qwen2.5-VL/Qwen3-VL v4-compatible iterative zoom tool."""

import json
import logging
import math
import re
from typing import Any, Optional
from uuid import uuid4

from PIL import Image
from qwen_vl_utils import fetch_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class ImgSurfZoomTool(BaseTool):
    """Refine a normalized Qwen-VL box with the active policy, then return its crop.

    The inner generations use the same rollout server/weights as the parent agent,
    matching SCAgent's parameter-sharing subagent design. Inner tokens are an
    environment operation and therefore are not directly included in the PPO mask.
    """

    requires_llm = True

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict[str, Any]] = {}
        self.coordinate_scale = float(config.get("coordinate_scale", 1000.0))
        self.max_iter = int(config.get("max_iter", 4))
        self.expand_ratio = float(config.get("expand_ratio", 0.4))
        self.iou_threshold = float(config.get("iou_threshold", 0.5))
        self.min_pixels = int(config.get("min_pixels", 32 * 32))
        self.max_pixels = int(config.get("max_pixels", 4_194_304))
        self.resize_factor = int(config.get("resize_factor", 32))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        image = create_kwargs.get("image", kwargs.get("image"))
        if image is None:
            raise ValueError("image_zoom_in_tool requires create_kwargs.image")
        self._instances[instance_id] = {
            "image": fetch_image({"image": image}).convert("RGB"),
            "question": str(create_kwargs.get("question", "")),
        }
        return instance_id, ToolResponse()

    @staticmethod
    def _valid_box(box: Any) -> bool:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError):
            return False
        return all(math.isfinite(v) for v in (x1, y1, x2, y2)) and x1 < x2 and y1 < y2

    def _normalized_to_image(self, box: list[float], image_size: tuple[int, int]) -> list[float]:
        width, height = image_size
        x1, y1, x2, y2 = (float(value) for value in box)
        scale = self.coordinate_scale
        result = [
            max(0.0, min(width, x1 / scale * width)),
            max(0.0, min(height, y1 / scale * height)),
            max(0.0, min(width, x2 / scale * width)),
            max(0.0, min(height, y2 / scale * height)),
        ]
        if not self._valid_box(result):
            raise ValueError(f"invalid normalized bbox_2d: {box}")
        return result

    def _relative_to_image(self, box: list[float], window: list[float]) -> list[float]:
        x1, y1, x2, y2 = box
        wx1, wy1, wx2, wy2 = window
        width, height = wx2 - wx1, wy2 - wy1
        scale = self.coordinate_scale
        result = [
            wx1 + max(0.0, min(scale, x1)) / scale * width,
            wy1 + max(0.0, min(scale, y1)) / scale * height,
            wx1 + max(0.0, min(scale, x2)) / scale * width,
            wy1 + max(0.0, min(scale, y2)) / scale * height,
        ]
        if not self._valid_box(result):
            raise ValueError(f"invalid refined bbox_2d: {box}")
        return result

    def _expanded_window(self, box: list[float], image_size: tuple[int, int]) -> list[float]:
        """v4 quarter-mode expansion: a 0.4-image window enclosing the box."""
        width, height = image_size
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half_w = self.expand_ratio * width / 2.0
        half_h = self.expand_ratio * height / 2.0
        return [
            max(0.0, min(x1, cx - half_w)),
            max(0.0, min(y1, cy - half_h)),
            min(float(width), max(x2, cx + half_w)),
            min(float(height), max(y2, cy + half_h)),
        ]

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

    def _crop(self, image: Image.Image, box: list[float]) -> Image.Image:
        width, height = image.size
        int_box = (
            max(0, int(math.floor(box[0]))),
            max(0, int(math.floor(box[1]))),
            min(width, int(math.ceil(box[2]))),
            min(height, int(math.ceil(box[3]))),
        )
        crop = image.crop(int_box)
        crop_w, crop_h = crop.size
        pixels = max(1, crop_w * crop_h)
        scale = 1.0
        if pixels > self.max_pixels:
            scale = math.sqrt(self.max_pixels / pixels)
        elif pixels < self.min_pixels:
            scale = math.sqrt(self.min_pixels / pixels)
        new_w = max(self.resize_factor, round(crop_w * scale / self.resize_factor) * self.resize_factor)
        new_h = max(self.resize_factor, round(crop_h * scale / self.resize_factor) * self.resize_factor)
        if (new_w, new_h) != crop.size:
            crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return crop

    @staticmethod
    def _extract_refined_call(text: str) -> tuple[Optional[list[float]], str]:
        matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        for raw in matches:
            try:
                call = json.loads(raw)
                arguments = call.get("arguments", call)
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                box = arguments.get("bbox_2d", arguments.get("bbox"))
                if ImgSurfZoomTool._valid_box(box):
                    return [float(value) for value in box], str(arguments.get("label", ""))
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                continue
        return None, ""

    async def _refine_once(
        self,
        crop: Image.Image,
        question: str,
        label: str,
        toolkit: dict[str, Any],
    ) -> tuple[Optional[list[float]], str]:
        processor = toolkit["processor"]
        server_manager = toolkit["llm_server_manager"]
        sampling_params = dict(toolkit["sampling_params"])
        apply_kwargs = dict(toolkit.get("apply_chat_template_kwargs") or {})
        messages = [
            {
                "role": "system",
                "content": (
                    "You refine a region in a cropped image. Locate the smallest region that contains the visual "
                    "evidence needed for the question. Coordinates are normalized to [0, 1000] relative to this "
                    "crop. Return exactly one image_zoom_in_tool call and no final answer."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            f"Question: {question}\nTarget label: {label or 'relevant evidence'}\n"
                            "Return <tool_call>{\"name\":\"image_zoom_in_tool\",\"arguments\":"
                            "{\"bbox_2d\":[x1,y1,x2,y2],\"label\":\"...\"}}</tool_call>."
                        ),
                    },
                ],
            },
        ]
        raw_prompt = processor.apply_chat_template(
            messages,
            tools=[self.tool_schema.model_dump(exclude_none=True, exclude_unset=True)],
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        model_inputs = processor(text=[raw_prompt], images=[crop], return_tensors="pt")
        prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
        response_ids = await server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=[crop],
        )
        text = processor.decode(response_ids, skip_special_tokens=False)
        return self._extract_refined_call(text)

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        instance = self._instances[instance_id]
        image: Image.Image = instance["image"]
        question = instance["question"]
        raw_box = parameters.get("bbox_2d", parameters.get("bbox"))
        label = str(parameters.get("label", ""))
        toolkit = parameters.get("llm_calling_toolkit")
        if not self._valid_box(raw_box):
            return ToolResponse(text="Error: bbox_2d must contain four ordered numbers."), -0.05, {"success": False}
        if not toolkit:
            return ToolResponse(text="Error: iterative refiner is unavailable."), -0.05, {"success": False}

        try:
            current = self._normalized_to_image([float(value) for value in raw_box], image.size)
            final_box = current
            iterations = 0
            converged = False
            for _ in range(self.max_iter):
                window = self._expanded_window(current, image.size)
                refined_normalized, refined_label = await self._refine_once(
                    self._crop(image, window), question, label, toolkit
                )
                if refined_normalized is None:
                    break
                refined = self._relative_to_image(refined_normalized, window)
                iterations += 1
                final_box = self._union(current, refined)
                if refined_label:
                    label = refined_label
                if self._iou(current, refined) > self.iou_threshold:
                    converged = True
                    break
                current = refined

            output_crop = self._crop(image, final_box)
            rounded = [int(round(value)) for value in final_box]
            text = (
                f"Refined crop for {label or 'relevant evidence'}; original-image bbox={rounded}; "
                f"iterations={iterations}; converged={str(converged).lower()}."
            )
            # ToolResponse uses a list for multimodal payloads even when the
            # tool returns only one image.
            return ToolResponse(text=text, image=[output_crop]), 0.0, {
                "success": True,
                "iterations": iterations,
                "converged": converged,
            }
        except Exception as exc:
            logger.warning("ImgSurf zoom refinement failed: %s", exc, exc_info=True)
            return ToolResponse(text=f"Error: zoom refinement failed: {exc}"), -0.05, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
