"""SCoT evaluation on the V* benchmark for Qwen2.5-VL and Qwen3-VL.

The main-agent/subagent prompt and rollout are shared with
``sCoT-main/eval/eval_subagent.py``.  Results use the JSONL schema consumed by
``eval/deepeyes/judge_result_vstar.py``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

from openai import OpenAI
from PIL import Image
import requests
from tqdm import tqdm


multiprocessing.set_start_method("spawn", force=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scot_eval_common import (  # noqa: E402
    make_question_prompt,
    prepare_image,
    run_subagent_conversation,
)


parser = argparse.ArgumentParser(description="Run SCoT on the V* benchmark.")
parser.add_argument("--model_name", type=str, default="qwen", help="Model name used for result save and inference")
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument("--api_url", type=str, default="http://localhost:18901/v1", help="OpenAI-compatible API URL")
parser.add_argument("--vstar_bench_path", type=str, required=True, help="Path to the V* benchmark")
parser.add_argument("--save_path", type=str, required=True, help="Root directory for JSONL results")
parser.add_argument("--eval_model_name", type=str, default=None, help="Served model name; defaults to --model_name")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--qwen_ver", type=int, default=3, choices=[2, 3], help="Qwen-VL version")
args = parser.parse_args()


client = OpenAI(api_key=args.api_key, base_url=args.api_url)
if args.eval_model_name is None:
    try:
        model_response = requests.get(f"{args.api_url.rstrip('/')}/models", timeout=30)
        model_response.raise_for_status()
        inference_model_name = model_response.json()["data"][0]["id"]
    except Exception:
        inference_model_name = args.model_name
else:
    inference_model_name = args.eval_model_name

ABC_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}
SAVE_ROOT = os.path.join(args.save_path, args.model_name)
os.makedirs(SAVE_ROOT, exist_ok=True)


def _annotation_path(image_path: str) -> str:
    path = Path(image_path)
    return str(path.with_suffix(".json"))


def _options_text(options: list[str]) -> str:
    return "\n" + "".join(f"{ABC_MAP[index + 1]}. {option}\n" for index, option in enumerate(options))


def process(image_arg: tuple[str, str]) -> dict:
    image_name, test_path = image_arg
    image_path = os.path.join(test_path, image_name)
    with open(_annotation_path(image_path), "r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    question = str(annotation["question"])
    options = [str(option) for option in annotation["options"]]
    prompt = make_question_prompt(question, _options_text(options), args.qwen_ver)
    image = Image.open(image_path).convert("RGB")
    image_info = prepare_image(image, args.qwen_ver)
    result = run_subagent_conversation(
        client=client,
        model_name=inference_model_name,
        image_info=image_info,
        prompt=prompt,
    )
    # V* annotations encode the correct answer as the first option.  This is
    # the same convention used by eval_vstar_v0.py and its judge script.
    return {
        "image": image_name,
        "question": question,
        "answer": options[0],
        "pred_ans": result["pred_ans"],
        "pred_output": result["pred_output"],
        "status": result["status"],
        "function_call_count": result["function_call_count"],
    }


def run_test_type(test_type: str) -> None:
    test_path = os.path.join(args.vstar_bench_path, test_type)
    image_files = sorted(
        file_name
        for file_name in os.listdir(test_path)
        if Path(file_name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    image_args = [(image_name, test_path) for image_name in image_files]
    save_name = f"result_{test_type}_{args.model_name}.jsonl"
    save_path = os.path.join(SAVE_ROOT, save_name)

    save_json: list[dict] = []
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(image_args), desc=f"Processing V* {test_type}") as progress:
            for result in pool.imap(process, image_args):
                if result is not None:
                    save_json.append(result)
                progress.update(1)
    with open(save_path, "w", encoding="utf-8") as handle:
        for item in save_json:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    for _test_type in ("direct_attributes", "relative_position"):
        run_test_type(_test_type)
