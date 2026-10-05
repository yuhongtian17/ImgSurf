"""SCoT evaluation on HRBench for Qwen2.5-VL and Qwen3-VL.

The main-agent/subagent prompt and rollout are shared with
``sCoT-main/eval/eval_subagent.py``.  Results use the JSONL schema consumed by
``eval/deepeyes/judge_result_hrbench.py``.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI
import requests
from tqdm import tqdm


multiprocessing.set_start_method("spawn", force=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scot_eval_common import make_question_prompt, prepare_image, run_subagent_conversation  # noqa: E402


parser = argparse.ArgumentParser(description="Run SCoT on HRBench.")
parser.add_argument("--model_name", type=str, default="qwen", help="Model name used for result save and inference")
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument("--api_url", type=str, default="http://localhost:18901/v1", help="OpenAI-compatible API URL")
parser.add_argument("--hrbench_path", type=str, required=True, help="Path to the HRBench directory")
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
SAVE_ROOT = os.path.join(args.save_path, args.model_name)
os.makedirs(SAVE_ROOT, exist_ok=True)


def _decode_image(value: str):
    import base64
    import io

    from PIL import Image

    image_data = base64.b64decode(value)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def process(item: tuple[int, dict]) -> dict:
    index, row = item
    image = _decode_image(str(row["image"]))
    question = str(row["question"])
    answer = str(row["answer"])
    answer_str = str(row[answer])
    options = [f"{letter}. {row[letter]}" for letter in ("A", "B", "C", "D")]
    category = str(row["category"])
    prompt = make_question_prompt(question, "\n" + "\n".join(options) + "\n", args.qwen_ver)
    image_info = prepare_image(image, args.qwen_ver)
    result = run_subagent_conversation(
        client=client,
        model_name=inference_model_name,
        image_info=image_info,
        prompt=prompt,
    )
    return {
        "index": index,
        "question": question,
        "answer": answer,
        "answer_str": answer_str,
        "pred_ans": result["pred_ans"],
        "pred_output": result["pred_output"],
        "category": category,
        "status": result["status"],
        "function_call_count": result["function_call_count"],
    }


def run_test_type(test_type: str) -> None:
    tsv_path = os.path.join(args.hrbench_path, f"{test_type}.tsv")
    dataframe = pd.read_csv(tsv_path, sep="\t")
    items = [(index, dataframe.iloc[index].to_dict()) for index in range(len(dataframe))]
    save_name = f"result_{test_type}_{args.model_name}.jsonl"
    save_path = os.path.join(SAVE_ROOT, save_name)

    save_json: list[dict] = []
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(items), desc=f"Processing HRBench {test_type}") as progress:
            for result in pool.imap(process, items):
                if result is not None:
                    save_json.append(result)
                progress.update(1)
    import json

    with open(save_path, "w", encoding="utf-8") as handle:
        for item in save_json:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    for _test_type in ("hr_bench_4k", "hr_bench_8k"):
        run_test_type(_test_type)
