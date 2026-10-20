"""Shared SCoT evaluation helpers for V* and HRBench.

The conversation protocol intentionally follows ``sCoT-main/eval/eval_subagent.py``:
the main VLM may call ``vlm_subagent_tool`` and the same served model answers the
subtask before the result is appended as a ``<tool_response>``.
"""

from __future__ import annotations

import ast
import base64
import json
import math
import re
import traceback
from io import BytesIO
from typing import Any

from PIL import Image


TOOL_NAME = "vlm_subagent_tool"
TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "The prompt to be passed to the VLM model. Wisely design the prompt, "
                "so you can make it into any kind oftool on downstream tasks of VLM."
            ),
        },
        "img_idx": {
            "type": "number",
            "description": "The index of the image (starting from 0) in the messages to be analyzed.",
        },
        "task_type": {
            "type": "string",
            "description": (
                "The type of the task you want to perform. For example, \"full_ocr\", "
                "\"grounding\", \"subregion_caption\", \"subregion_ocr\", and "
                "\"subregion_question_answering\". If this is a subregion task, you "
                "must provide `bbox_2d` parameter."
            ),
        },
        "bbox_2d": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": (
                "The bounding box of the region if you want to zoom in, as "
                "[x1, y1, x2, y2] (left, top, right, bottom). If you want to perform "
                "a subregion task, you must provide `bbox_2d`."
            ),
        },
    },
    "required": ["prompt", "img_idx", "task_type"],
}
TOOL_DESC = """
USE THIS SUBAGENT TOOL TO SOLVE SUBTASKS TO HELP YOU COMPLETE THE TASK.

This tool is a vision-language model (VLM) model.
You can make it into any kind of tool on downstream tasks of VLM by carefully designing the prompt.
Possible usages include OCR, Caption, Reasoning, etc.
"""
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESC,
        "parameters": TOOL_PARAMS,
    },
}
TOOL_STRING = f"<tools>\n{json.dumps(TOOL_SCHEMA)}\n</tools>"


# These prompts are copied from sCoT-main/eval/eval_subagent.py so that the
# benchmark scripts exercise the same tool-use policy and subagent protocol.
INSTRUCTION_PROMPT_SYSTEM = f"""You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
{TOOL_STRING}

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

## Example Usages
1. Full Image OCR: Output only the text content from the image without any additional descriptions or formatting.
    ```json
        <tool_call>
        {{
            "name": "vlm_subagent_tool",
            "arguments": {{
                "prompt": "Please output only the text content from the image without any additional descriptions or formatting.",
                "img_idx": 0,
                "task_type": "ocr",
            }}
        }}
        </tool_call>
    ```
    - Use it when you want to know all the texts in the image
2. Visual Grounding: Locate the objects in the image and return the bbox_2d.
    - Usage: Use it when you want to find the hardly visible objects in the image.
        ```json
            <tool_call>
            {{
                "name": "vlm_subagent_tool",
                "arguments": {{
                    "prompt": "Outline the position of each object and output all the coordinates in JSON format.",
                    "img_idx": 0,
                    "task_type": "grounding",
                }}
            }}
            </tool_call>
        ```
3. Subregion Caption: Describe the image in a few sentences. Must provide `bbox_2d` to zoom in on the region.
    - Usage: Use it when you want to take a close look at a specific region in the image.
        ```json
            <tool_call>
            {{
                "name": "vlm_subagent_tool",
                "arguments": {{
                    "prompt": "Please describe the image in a few sentences.",
                    "img_idx": 0,
                    "task_type": "subregion_caption",
                    "bbox_2d": [63, 32, 150, 170]
                }}
            }}
            </tool_call>
        ```
4. Subregion OCR: Read out the text in the region located by the bbox_2d. Must provide `bbox_2d` to zoom in on the region.
    - Usage: Use it when you want to know the text in a specific region in the image.
        ```json
            <tool_call>
            {{
                "name": "vlm_subagent_tool",
                "arguments": {{
                    "prompt": "Outline the position of each text and output all the coordinates in JSON format.",
                    "img_idx": 0,
                    "task_type": "subregion_ocr",
                    "bbox_2d": [63, 32, 150, 170]
                }}
            }}
            </tool_call>
        ```
"""

USER_PROMPT_V2 = (
    "\nCall **vlm_subagent_tool** if needed, then answer. Format strictly as: "
    "<tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
)
QWEN3_COORDINATE_HINT = (
    "\nCoordinate convention for Qwen3-VL: whenever you output `bbox_2d`, "
    "use normalized coordinates in the [0, 1000] x [0, 1000] coordinate "
    "system of the image, ordered as [x1, y1, x2, y2]."
)
START_TOKEN = "<tool_call>"
END_TOKEN = "</tool_call>"


def coordinate_hint(qwen_ver: int) -> str:
    return QWEN3_COORDINATE_HINT if int(qwen_ver) == 3 else ""


def make_system_prompt(qwen_ver: int) -> str:
    return INSTRUCTION_PROMPT_SYSTEM + coordinate_hint(qwen_ver)


def make_question_prompt(question: str, options: str, qwen_ver: int) -> str:
    return f"Question: {question}\nOptions: {options}\n{coordinate_hint(qwen_ver)}{USER_PROMPT_V2}"


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def image_limits(qwen_ver: int) -> tuple[int, int, int]:
    if int(qwen_ver) not in {2, 3}:
        raise ValueError("--qwen_ver must be 2 or 3")
    factor = 32 if int(qwen_ver) == 3 else 28
    return factor, (2 * factor) ** 2, (128 * factor) ** 2


def smart_resize(
    width: int,
    height: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / max(1, height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return w_bar, h_bar


def encode_pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def prepare_image(image: Image.Image, qwen_ver: int) -> dict[str, Any]:
    """Resize an input and return the coordinate conversion used by the API."""

    factor, min_pixels, max_pixels = image_limits(qwen_ver)
    original = image.convert("RGB")
    original_width, original_height = original.size
    resized_width, resized_height = smart_resize(
        original_width, original_height, factor, min_pixels, max_pixels
    )
    resized = original.resize((resized_width, resized_height), resample=Image.BICUBIC)
    if int(qwen_ver) == 3:
        coordinate_width, coordinate_height = 1000.0, 1000.0
    else:
        coordinate_width, coordinate_height = float(resized_width), float(resized_height)
    return {
        "original": original,
        "resized": resized,
        "qwen_ver": int(qwen_ver),
        "factor": factor,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "original_size": (original_width, original_height),
        "resized_size": (resized_width, resized_height),
        "coordinate_size": (coordinate_width, coordinate_height),
        "base64_image": encode_pil_image_to_base64(resized),
    }


def parse_tool_call(raw_text: str) -> dict[str, Any]:
    """Parse JSON first, then accept the Python-literal form emitted by Qwen."""

    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        value = ast.literal_eval(raw_text)
    if not isinstance(value, dict):
        raise ValueError("tool call must be an object")
    return value


def _finite_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_2d must contain four numbers")
    bbox = [float(item) for item in value]
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError("bbox_2d must contain finite numbers")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("bbox_2d must be ordered left, top, right, bottom")
    return bbox


def bbox_to_crop(
    bbox: Any,
    image_info: dict[str, Any],
    interpolation_factor: float = 0.3,
) -> tuple[list[float], tuple[int, int, int, int]]:
    """Expand a model bbox and map it into the version-specific crop frame."""

    bbox_values = _finite_bbox(bbox)
    coordinate_width, coordinate_height = image_info["coordinate_size"]
    bbox_values = [
        max(0.0, min(coordinate_width, bbox_values[0])),
        max(0.0, min(coordinate_height, bbox_values[1])),
        max(0.0, min(coordinate_width, bbox_values[2])),
        max(0.0, min(coordinate_height, bbox_values[3])),
    ]
    if bbox_values[0] >= bbox_values[2] or bbox_values[1] >= bbox_values[3]:
        raise ValueError("bbox_2d is outside the image coordinate range")
    full = [0.0, 0.0, coordinate_width, coordinate_height]
    expanded = [
        interpolation_factor * bbox_values[index] + (1.0 - interpolation_factor) * full[index]
        for index in range(4)
    ]
    # Qwen2.5-VL follows sCoT's resized-image coordinate frame. Qwen3-VL is
    # explicitly prompted to emit [0, 1000] coordinates and the original
    # DeepEyes evaluators map those coordinates back to the original image.
    if int(image_info.get("qwen_ver", 2)) == 3:
        target_width, target_height = image_info["original_size"]
    else:
        target_width, target_height = image_info["resized_size"]
    left = max(0, min(int(expanded[0] / coordinate_width * target_width), target_width - 1))
    top = max(0, min(int(expanded[1] / coordinate_height * target_height), target_height - 1))
    right = max(left + 1, min(int(expanded[2] / coordinate_width * target_width), target_width))
    bottom = max(top + 1, min(int(expanded[3] / coordinate_height * target_height), target_height))
    return bbox_values, (left, top, right, bottom)


def _backup_tool_parser(tool_call_str: str) -> tuple[str, str]:
    pattern = r'"prompt"\s*:\s*"([^"\u0000]*?(?:\\.[^"\u0000]*?)*)"'
    prompt = re.findall(pattern, tool_call_str, re.DOTALL)
    if len(prompt) != 1:
        raise ValueError(f"expected one prompt in tool call, got {len(prompt)}")
    cleaned = re.sub(pattern, '"prompt": ""', tool_call_str, flags=re.DOTALL)
    return prompt[0], cleaned


def parse_subagent_action(raw_text: str) -> dict[str, Any]:
    """Parse the tool block, recovering escaped prompts like eval_subagent.py."""

    block = raw_text.split(START_TOKEN, 1)[1].split(END_TOKEN, 1)[0].strip()
    try:
        return parse_tool_call(block)
    except Exception:
        prompt, cleaned = _backup_tool_parser(block)
        action = parse_tool_call(cleaned)
        arguments = action.setdefault("arguments", {})
        if isinstance(arguments, dict):
            arguments["prompt"] = prompt
        return action


def _response_content(response: Any, stage: str) -> str:
    """Extract assistant text and turn malformed API replies into actionable errors."""

    if isinstance(response, dict):
        choices = response.get("choices")
    else:
        choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError(
            f"{stage}: API response has no choices; response={str(response)[:2000]}"
        )
    choice = choices[0]
    if isinstance(choice, dict):
        message = choice.get("message")
    else:
        message = getattr(choice, "message", None)
    if message is None:
        raise RuntimeError(
            f"{stage}: API response choice has no message; response={str(response)[:2000]}"
        )
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if content is None:
        # Some OpenAI-compatible servers return structured content blocks.
        if isinstance(message, dict):
            blocks = message.get("content_blocks")
        else:
            blocks = getattr(message, "content_blocks", None)
        if blocks:
            content = "".join(
                str(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", ""))
                for block in blocks
            )
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(
            str(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", ""))
            for block in content
        )
    return str(content)


def _crop_for_subagent(image_info: dict[str, Any], bbox: Any) -> tuple[Image.Image, list[float] | None]:
    if bbox is None:
        return image_info["resized"], None
    normalized, crop_box = bbox_to_crop(bbox, image_info)
    source = image_info["original"] if int(image_info.get("qwen_ver", 2)) == 3 else image_info["resized"]
    cropped = source.crop(crop_box)
    width, height = smart_resize(
        cropped.width,
        cropped.height,
        image_info["factor"],
        image_info["min_pixels"],
        image_info["max_pixels"],
    )
    return cropped.resize((width, height), resample=Image.BICUBIC), normalized


def run_subagent_conversation(
    *,
    client: Any,
    model_name: str,
    image_info: dict[str, Any],
    prompt: str,
    max_turns: int = 10,
) -> dict[str, Any]:
    """Run the SCoT main-agent/subagent loop and return judge-compatible fields."""

    base64_image = image_info["base64_image"]
    # Qwen3-VL's chat template has a dedicated ``tool`` branch that also
    # preserves multimodal content.  Qwen2.5-VL uses the generic role path,
    # so retain the ``function`` role used by eval_subagent.py there.
    tool_role = "tool" if int(image_info.get("qwen_ver", 2)) == 3 else "function"
    system_prompt = make_system_prompt(int(image_info.get("qwen_ver", 2)))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                # encode_pil_image_to_base64 writes PNG bytes, so keep the
                # data-URL media type consistent for vLLM's image loader.
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    "max_pixels": image_info["max_pixels"],
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    print_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    response_message = ""
    status = "success"
    function_call_count = 0
    try:
        for turn_idx in range(max_turns + 1):
            if "</answer>" in response_message:
                break
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=10240,
                stop=["<|im_end|>".strip(), END_TOKEN],
            )
            response_message = _response_content(response, "main-agent")
            if START_TOKEN not in response_message:
                messages.append({"role": "assistant", "content": response_message})
                print_messages.append({"role": "assistant", "content": response_message})
                continue

            response_message = response_message.replace("addCriterion", "")
            function_call_count += 1
            action = parse_subagent_action(response_message)
            arguments = action.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError("vlm_subagent_tool arguments must be an object")
            task_type = str(arguments.get("task_type", "subregion_question_answering"))
            subagent_prompt = str(arguments.get("prompt", ""))
            bbox = arguments.get("bbox_2d")
            crop, _ = _crop_for_subagent(image_info, bbox)
            crop_base64 = encode_pil_image_to_base64(crop)
            subagent_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": tool_role,
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{crop_base64}"},
                            "max_pixels": image_info["max_pixels"],
                        },
                        {"type": "text", "text": f"[{task_type}] {subagent_prompt}"},
                    ],
                },
            ]
            subagent_response = client.chat.completions.create(
                model=model_name,
                messages=subagent_messages,
                temperature=0.0,
                max_tokens=10240,
                stop=["<|im_end|>".strip()],
            )
            subagent_content = _response_content(subagent_response, "subagent")
            # Keep the model-emitted coordinates in the visible trajectory,
            # matching eval_subagent.py; normalization is only for cropping.
            queried_bbox = bbox
            tool_response_body = [
                {"type": "text", "text": f"[Queried at Bounding Box {queried_bbox} of the original image.]"},
                {"type": "text", "text": subagent_content},
            ]
            # Qwen3-VL's ``tool`` chat-template branch adds the enclosing
            # <tool_response> tags itself. Qwen2.5-VL follows the generic
            # role path, so it needs the explicit tags used by eval_subagent.
            tool_response_for_api = tool_response_body if tool_role == "tool" else [
                {"type": "text", "text": "<tool_response>"},
                *tool_response_body,
                {"type": "text", "text": "</tool_response>"},
            ]
            tool_response_for_print = [
                {"type": "text", "text": "<tool_response>"},
                *tool_response_body,
                {"type": "text", "text": "</tool_response>"},
            ]
            messages.extend(
                [
                    {"role": "assistant", "content": response_message},
                    {"role": tool_role, "content": tool_response_for_api},
                ]
            )
            print_messages.extend(
                [
                    {"role": "assistant", "content": response_message},
                    {"role": tool_role, "content": tool_response_for_print},
                ]
            )
        else:
            status = "max_turns"
    except Exception as exc:
        print(f"Error!!!! {type(exc).__name__}: {exc}")
        traceback.print_exc()
        status = "error"

    if "<answer>" in response_message and "</answer>" in response_message:
        pred_ans = response_message.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    else:
        pred_ans = response_message
    return {
        "pred_ans": pred_ans,
        "pred_output": print_messages,
        "status": status,
        "function_call_count": function_call_count,
    }
