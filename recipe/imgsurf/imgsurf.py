# Copyright 2026 ImgSurf contributors
# Licensed under the Apache License, Version 2.0
"""Dataset adapter and hybrid reward for Qwen2.5-VL/Qwen3-VL ImgSurf GRPO."""

import io
import logging
import math
import os
import re
import string
from functools import lru_cache
from typing import Any

from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from recipe.imgsurf.prompt_utils import SYSTEM_PROMPT, render_imgsurf_chat_prompt

logger = logging.getLogger(__name__)


def _processor_family(processor) -> str:
    """Return the supported Qwen-VL processor family without using model paths."""
    name = processor.image_processor.__class__.__name__
    if "Qwen3VLImageProcessor" in name:
        return "qwen3_vl"
    # Qwen2.5-VL intentionally reuses Qwen2VLImageProcessor in Transformers.
    if "Qwen2VLImageProcessor" in name:
        return "qwen2_5_vl"
    raise ValueError(
        "ImgSurf supports only Qwen2.5-VL and Qwen3-VL processors; "
        f"received {name}."
    )


def _resize_to_pixel_budget(image: Image.Image, max_pixels: int) -> Image.Image:
    image = image.convert("RGB")
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    size = (max(32, int(image.width * scale)), max(32, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


class CustomRLHFDataset(RLHFDataset):
    """Turn the three DeepEyes parquet subsets into one Qwen-VL tool task."""

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        extra_info = row_dict.get("extra_info") or {}
        question = str(extra_info.get("question") or self._fallback_question(row_dict))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            question
                            + "\nThink first; use image_zoom_in_tool only if it improves reliability; "
                            "then put the final response in <answer>...</answer>."
                        ),
                    },
                ],
            },
        ]

        max_pixels = int(os.environ.get("IMGSURF_MAX_INPUT_PIXELS", "4194304"))
        raw_images = row_dict.pop(self.image_key, None) or []
        images = []
        for image in raw_images:
            if image.get("bytes") is not None:
                decoded = Image.open(io.BytesIO(image["bytes"]))
            elif image.get("path"):
                decoded = Image.open(image["path"])
            else:
                raise ValueError("DeepEyes image entry contains neither bytes nor path")
            images.append(_resize_to_pixel_budget(decoded, max_pixels))
        if len(images) != 1:
            raise ValueError(f"ImgSurf currently expects exactly one image per sample, got {len(images)}")

        raw_prompt = render_imgsurf_chat_prompt(
            self.processor,
            messages,
            self.apply_chat_template_kwargs,
        )
        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        model_inputs.pop("second_per_grid_ts", None)

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        processor_family = _processor_family(self.processor)
        if processor_family == "qwen3_vl":
            from verl.models.transformers.qwen3_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            ).unsqueeze(0)
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            ).unsqueeze(0)
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["multi_modal_data"] = {"image": images}

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        row_dict["index"] = extra_info.get("index", item)
        row_dict["tools_kwargs"] = {
            "image_zoom_in_tool": {
                "create_kwargs": {"image": images[0], "question": question},
            }
        }
        row_dict["agent_name"] = "imgsurf_tool_agent"
        return row_dict

    def _fallback_question(self, row_dict: dict) -> str:
        prompt = row_dict.get(self.prompt_key) or []
        if len(prompt) < 2:
            return "Answer the question shown in the image."
        text = str(prompt[-1].get("content", "")).replace("<image>", "").strip()
        return re.split(r"\nThink first", text, maxsplit=1)[0].strip()


def _extract_answer(solution: str) -> tuple[str, bool]:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", solution, flags=re.DOTALL | re.IGNORECASE)
    balanced = solution.lower().count("<answer>") == solution.lower().count("</answer>") == 1
    answer = matches[0].strip() if len(matches) == 1 else ""
    if not answer:
        # Keep a weak fallback for accuracy logging, but it never earns format reward.
        tail = re.sub(r"<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>", "", solution, flags=re.DOTALL)
        answer = tail.split("</think>")[-1].strip()
    return answer, bool(balanced and len(matches) == 1 and answer)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans({char: " " for char in string.punctuation if char not in ".-/"}))
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return " ".join(text.split())


def _choice(text: str) -> str | None:
    match = re.fullmatch(r"\s*([A-Da-d])\s*[.)]?\s*", text)
    return match.group(1).upper() if match else None


def _polarity(text: str) -> str | None:
    normalized = _normalize(text)
    if re.match(r"^(yes|true|correct)\b", normalized):
        return "yes"
    if re.match(r"^(no|false|incorrect)\b", normalized) or " not " in f" {normalized} ":
        return "no"
    return None


def _rule_semantic_match(prediction: str, ground_truth: str) -> bool | None:
    pred, gold = _normalize(prediction), _normalize(ground_truth)
    if not pred:
        return False
    if pred == gold:
        return True
    gold_choice = _choice(ground_truth)
    if gold_choice:
        pred_choice = re.match(r"\s*([A-Da-d])(?:\s*[.):]|\s*$)", prediction)
        return bool(pred_choice and pred_choice.group(1).upper() == gold_choice)
    gold_polarity = _polarity(ground_truth)
    pred_polarity = _polarity(prediction)
    if gold_polarity and pred_polarity:
        return gold_polarity == pred_polarity

    gold_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", gold)
    pred_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", pred)
    if gold_numbers and pred_numbers and gold_numbers == pred_numbers:
        return True

    direction_words = {"left", "right", "above", "below", "top", "bottom", "inside", "outside", "front", "behind"}
    gold_directions = set(gold.split()) & direction_words
    pred_directions = set(pred.split()) & direction_words
    if gold_directions:
        return gold_directions == pred_directions

    # A short answer contained in the reference is common in VStar ("tan" vs a sentence).
    if len(pred.split()) <= 5 and re.search(rf"\b{re.escape(pred)}\b", gold):
        return True
    pred_tokens, gold_tokens = set(pred.split()), set(gold.split())
    if pred_tokens and gold_tokens:
        precision = len(pred_tokens & gold_tokens) / len(pred_tokens)
        recall = len(pred_tokens & gold_tokens) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 >= 0.8:
            return True
    return None


def _math_match(prediction: str, ground_truth: str) -> bool:
    try:
        from verl.utils.reward_score.math_verify import compute_score as math_verify_score

        if bool(math_verify_score(prediction, ground_truth)):
            return True
    except Exception as exc:
        logger.debug("math-verify failed, using normalized fallback: %s", exc)
    pred = _normalize(prediction.replace("\\boxed", ""))
    gold = _normalize(ground_truth.replace("\\boxed", ""))
    return pred == gold or pred.rstrip(".0") == gold.rstrip(".0")


@lru_cache(maxsize=8192)
def _judge_semantic(question: str, ground_truth: str, prediction: str) -> bool:
    base_url = os.environ.get("IMGSURF_JUDGE_BASE_URL", "").strip()
    if not base_url:
        return False
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("IMGSURF_JUDGE_API_KEY", "EMPTY"),
            timeout=float(os.environ.get("IMGSURF_JUDGE_TIMEOUT", "20")),
        )
        model = os.environ.get("IMGSURF_JUDGE_MODEL", "").strip()
        if not model:
            model = client.models.list().data[0].id
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=4,
            messages=[
                {
                    "role": "system",
                    "content": "Judge semantic answer equivalence strictly. Reply only CORRECT or INCORRECT.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nReference: {ground_truth}\nCandidate: {prediction}",
                },
            ],
        )
        return response.choices[0].message.content.strip().upper().startswith("CORRECT")
    except Exception as exc:
        logger.warning("Optional ImgSurf judge failed: %s", exc)
        return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Accuracy-first hybrid reward with small format/tool shaping terms.

    score = accuracy + 0.10*format + 0.10*correct_visual_tool
            - 0.05*invalid_tool - 0.02*excess_tool, clipped to [-0.2, 1.2].
    """
    extra_info = extra_info or {}
    answer, answer_format_ok = _extract_answer(solution_str)
    tag_format_ok = (
        solution_str.count("<think>") == solution_str.count("</think>")
        and solution_str.count("<tool_call>") == solution_str.count("</tool_call>")
        and len(answer) <= 1000
    )
    format_raw = 1.0 if answer_format_ok and tag_format_ok else -1.0

    source = str(data_source).lower()
    if "thinklite" in source or str(extra_info.get("ability", "")).lower() == "math":
        accuracy = float(_math_match(answer, str(ground_truth)))
        visual_source = False
    else:
        rule_result = _rule_semantic_match(answer, str(ground_truth))
        if rule_result is None:
            rule_result = _judge_semantic(
                str(extra_info.get("question", "")), str(ground_truth), answer
            )
        accuracy = float(bool(rule_result))
        visual_source = True

    tool_calls = len(re.findall(r"<tool_call>.*?</tool_call>", solution_str, flags=re.DOTALL))
    invalid_tools = len(re.findall(r"<tool_response>\s*Error:", solution_str, flags=re.DOTALL | re.IGNORECASE))
    valid_tool = float(tool_calls > invalid_tools and tool_calls > 0)
    correct_tool_bonus = float(visual_source and accuracy > 0.5 and valid_tool > 0.5)
    excess_tools = max(0, tool_calls - 2)
    score = accuracy + 0.10 * format_raw + 0.10 * correct_tool_bonus - 0.05 * invalid_tools - 0.02 * excess_tools
    score = max(-0.2, min(1.2, score))
    return {
        "score": float(score),
        "accuracy": accuracy,
        "format": format_raw,
        "correct_tool_bonus": correct_tool_bonus,
        "tool_calls": float(tool_calls),
        "invalid_tools": float(invalid_tools),
    }


if __name__ == "__main__":
    assert compute_score("chart", "<think>x</think><answer>B</answer>", "B")["score"] == 1.1
    assert compute_score("vstar", "<answer>No.</answer>", "No, the car is not left.")["accuracy"] == 1.0
    assert compute_score("thinklite_eureka", "<answer>-4</answer>", "-4")["accuracy"] == 1.0
    print("ImgSurf reward smoke tests passed")
