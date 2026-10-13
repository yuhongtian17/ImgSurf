#!/usr/bin/env python3
"""Extract HR-Bench evaluation records, images, and tool crops.

The evaluator writes one JSON object per HR-Bench question.  This utility joins
those objects with the corresponding TSV rows, writes de-duplicated source
images as numbered PNG files, exports the regions used by the model, and saves
one ``result.json`` per question.

The result root follows the layout used by ``eval_hrbench_v*.py``::

    <save_path>/<model_name>/result_hr_bench_4k_<model_name>_acc.jsonl
    <save_path>/<model_name>/extract_4k/

Examples::

    python eval/deepeyes_old/extract_hr_bench.py \
        --model_name qwen3-vl-4b \
        --hrbench_path /path/to/HR-Bench \
        --save_path /path/to/work_dirs/v2 \
        --mode 4k
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import csv
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - the evaluation environment normally has tqdm
    def tqdm(iterable: Iterable[Any] | None = None, **_: Any):
        return iterable if iterable is not None else []


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)(?:</tool_call>|$)", re.DOTALL)
REQUIRED_TSV_COLUMNS = {"answer", "question", "A", "B", "C", "D", "category", "image"}


@dataclass(frozen=True)
class SampleInfo:
    """Metadata for one TSV row and its de-duplicated image."""

    question_number: int
    image_number: int
    category: str
    question: str
    answer: str
    answer_str: str
    image_size: tuple[int, int]


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def decode_image_bytes(value: str, row_number: int) -> bytes:
    if not value:
        raise ValueError(f"TSV row {row_number}: image field is empty")
    if value.startswith("data:"):
        marker = ";base64,"
        if marker not in value:
            raise ValueError(f"TSV row {row_number}: unsupported image data URL")
        value = value.split(marker, 1)[1]
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"TSV row {row_number}: invalid base64 image") from exc


def _image_class():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the user's environment
        raise RuntimeError(
            "Pillow is required. Install it with `python -m pip install Pillow`."
        ) from exc
    return Image


def save_source_image(image_bytes: bytes, destination: Path) -> tuple[int, int]:
    """Save the source pixels as PNG without changing their coordinate frame."""

    Image = _image_class()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        # The evaluators crop the image in the orientation returned by PIL, so
        # do not apply EXIF transposition here.
        image = source.convert("RGB")
        image.save(destination, format="PNG")
        return image.size


def read_tsv_samples(tsv_path: Path, images_dir: Path) -> list[SampleInfo]:
    """Decode the TSV and write first-occurrence images as 001.png, 002.png, ..."""

    configure_csv_field_limit()
    images_dir.mkdir(parents=True, exist_ok=True)
    digest_to_number: dict[str, int] = {}
    digest_to_size: dict[str, tuple[int, int]] = {}
    samples: list[SampleInfo] = []

    with tsv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_TSV_COLUMNS - columns)
        if missing:
            raise ValueError(f"{tsv_path} is missing TSV columns: {', '.join(missing)}")

        for question_number, row in enumerate(
            tqdm(reader, desc=f"{images_dir.parent.name}/images", unit="row"), start=1
        ):
            image_bytes = decode_image_bytes(row["image"], question_number + 1)
            digest = hashlib.sha256(image_bytes).hexdigest()
            image_number = digest_to_number.get(digest)
            if image_number is None:
                image_number = len(digest_to_number) + 1
                digest_to_number[digest] = image_number
                image_path = images_dir / f"{image_number:03d}.png"
                image_size = save_source_image(image_bytes, image_path)
                digest_to_size[digest] = image_size
            else:
                image_size = digest_to_size[digest]

            answer = str(row["answer"]).strip()
            if answer not in {"A", "B", "C", "D"}:
                raise ValueError(
                    f"TSV row {question_number + 1}: answer must be A/B/C/D, got {answer!r}"
                )
            samples.append(
                SampleInfo(
                    question_number=question_number,
                    image_number=image_number,
                    category=str(row["category"]).strip(),
                    question=str(row["question"]),
                    answer=answer,
                    answer_str=str(row[answer]),
                    image_size=image_size,
                )
            )

    return samples


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL entry {path}:{line_number} is not an object")
            yield value


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return "" if content is None else str(content)


def assistant_messages(pred_output: Any) -> list[dict[str, Any]]:
    if not isinstance(pred_output, list):
        return []
    return [
        message
        for message in pred_output
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]


def count_num_tool(entry: dict[str, Any], is_v2_or_v4: bool) -> int:
    messages = assistant_messages(entry.get("pred_output"))
    if is_v2_or_v4:
        # v2/v4 appends one final answer assistant message after the tool loop.
        return max(0, len(messages) - 1)
    return sum("<tool_call>" in content_text(message.get("content")) for message in messages)


def parse_tool_call(content: str) -> dict[str, Any] | None:
    match = TOOL_CALL_RE.search(content)
    if match is None:
        return None
    block = match.group(1).strip()
    try:
        value = json.loads(block)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(block)
        except (SyntaxError, ValueError):
            # A model can append text after an otherwise valid JSON object.
            decoder = json.JSONDecoder()
            try:
                value, _ = decoder.raw_decode(block)
            except json.JSONDecodeError:
                return None
    return value if isinstance(value, dict) else None


def bbox_from_tool_call(call: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(call, dict):
        return None
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return None
    value = arguments.get("bbox_2d", arguments.get("bbox"))
    return normalize_bbox(value)


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    return bbox


def crop_bbox(
    image: Any,
    bbox: Any,
    destination: Path,
    *,
    coordinate_size: tuple[float, float] | None,
) -> bool:
    """Crop a bbox and save it; return False for malformed/empty regions."""

    bbox_values = normalize_bbox(bbox)
    if bbox_values is None:
        return False
    image_width, image_height = image.size
    if coordinate_size is None:
        coordinate_width, coordinate_height = float(image_width), float(image_height)
    else:
        coordinate_width, coordinate_height = coordinate_size
    if coordinate_width <= 0 or coordinate_height <= 0:
        return False

    x1, y1, x2, y2 = bbox_values
    if coordinate_size is not None:
        x1 *= image_width / coordinate_width
        x2 *= image_width / coordinate_width
        y1 *= image_height / coordinate_height
        y2 *= image_height / coordinate_height
    left = max(0, min(int(math.floor(x1)), image_width - 1))
    top = max(0, min(int(math.floor(y1)), image_height - 1))
    right = max(left + 1, min(int(math.ceil(x2)), image_width))
    bottom = max(top + 1, min(int(math.ceil(y2)), image_height))
    if right <= left or bottom <= top:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.crop((left, top, right, bottom)).save(destination, format="PNG")
    return True


def extract_v0_crops(entry: dict[str, Any], image: Any, output_dir: Path) -> list[dict[str, Any]]:
    """Extract the first tool call from each v0 assistant tool-call message."""

    crops: list[dict[str, Any]] = []
    tool_index = 0
    for message in assistant_messages(entry.get("pred_output")):
        content = content_text(message.get("content"))
        if "<tool_call>" not in content:
            continue
        call = parse_tool_call(content)
        bbox = bbox_from_tool_call(call)
        tool_index += 1
        file_name = f"tool_call_{tool_index}.png"
        saved = crop_bbox(
            image,
            bbox,
            output_dir / file_name,
            coordinate_size=(1000.0, 1000.0),
        )
        crops.append({"file": file_name, "bbox_2d": bbox, "saved": saved})
    return crops


def region_bbox(region: Any) -> Any:
    if isinstance(region, dict):
        return region.get("bbox_2d", region.get("bbox"))
    return region


def extract_v2_crops(entry: dict[str, Any], image: Any, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    screened_outputs: list[dict[str, Any]] = []
    final_outputs: list[dict[str, Any]] = []
    for field_name, file_prefix, coordinate_size, destination in (
        ("screened_regions", "screened_region", (1000.0, 1000.0), screened_outputs),
        ("final_regions", "final_region", None, final_outputs),
    ):
        regions = entry.get(field_name)
        if not isinstance(regions, list):
            regions = []
        for index, region in enumerate(regions, start=1):
            bbox = normalize_bbox(region_bbox(region))
            file_name = f"{file_prefix}_{index}.png"
            saved = crop_bbox(
                image,
                bbox,
                output_dir / file_name,
                coordinate_size=coordinate_size,
            )
            destination.append({"file": file_name, "bbox_2d": bbox, "saved": saved})
    return screened_outputs, final_outputs


def locate_result_file(result_dir: Path, mode: str, model_name: str) -> Path:
    expected = result_dir / f"result_hr_bench_{mode}_{model_name}_acc.jsonl"
    if expected.is_file():
        return expected
    candidates = sorted(result_dir.glob(f"result_hr_bench_{mode}_*_acc.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No HR-Bench {mode} *_acc.jsonl found in {result_dir}")
    raise RuntimeError(
        f"Multiple HR-Bench {mode} *_acc.jsonl files found; expected {expected.name}"
    )


def write_summary(output_dir: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for category in ("single", "cross"):
        category_entries = [entry for entry in entries if entry.get("category") == category]
        total = len(category_entries)
        correct = sum(float(entry.get("acc", 0.0)) == 1.0 for entry in category_entries)
        tool_total = sum(int(entry.get("num_tool", 0)) for entry in category_entries)
        summary[category] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "average_num_tool": tool_total / total if total else 0.0,
        }
    summary["overall"] = {
        "total": len(entries),
        "correct": sum(float(entry.get("acc", 0.0)) == 1.0 for entry in entries),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def extract_mode(
    *,
    mode: str,
    model_name: str,
    hrbench_path: Path,
    result_dir: Path,
) -> None:
    output_dir = result_dir / f"extract_{mode}"
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = hrbench_path / f"hr_bench_{mode}.tsv"
    result_path = locate_result_file(result_dir, mode, model_name)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"HR-Bench TSV not found: {tsv_path}")

    samples = read_tsv_samples(tsv_path, images_dir)
    entries: list[dict[str, Any]] = []
    result_iterator = iter_jsonl(result_path)
    for question_number, sample in enumerate(
        tqdm(samples, desc=f"{output_dir.name}/entries", unit="entry"), start=1
    ):
        try:
            entry = next(result_iterator)
        except StopIteration as exc:
            raise ValueError(
                f"{result_path} has fewer entries than {tsv_path}: "
                f"missing question {question_number:03d}"
            ) from exc

        if entry.get("question") not in (None, sample.question):
            raise ValueError(
                f"Question order mismatch at {question_number:03d}: "
                f"TSV={sample.question!r}, JSONL={entry.get('question')!r}"
            )
        entry_category = entry.get("category")
        if entry_category not in (None, sample.category):
            raise ValueError(
                f"Category mismatch at {question_number:03d}: "
                f"TSV={sample.category!r}, JSONL={entry_category!r}"
            )

        question_dir = output_dir / (
            f"q{sample.question_number:03d}_i{sample.image_number:03d}_"
            f"{sample.category}_{int(float(entry.get('acc', 0.0)) == 1.0)}"
        )
        question_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / f"{sample.image_number:03d}.png"
        Image = _image_class()
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            is_v2_or_v4 = "screened_regions" in entry or "final_regions" in entry
            num_tool = count_num_tool(entry, is_v2_or_v4)
            if is_v2_or_v4:
                screened, final = extract_v2_crops(entry, image, question_dir)
                crop_info: dict[str, Any] = {
                    "screened_regions": screened,
                    "final_regions": final,
                }
            else:
                crop_info = {"tool_calls": extract_v0_crops(entry, image, question_dir)}

        result = dict(entry)
        result.update(
            {
                "question_number": sample.question_number,
                "image_number": sample.image_number,
                "image_file": f"../images/{sample.image_number:03d}.png",
                "num_tool": num_tool,
                "extract_version": "v2_or_v4" if is_v2_or_v4 else "v0",
                "crop_info": crop_info,
            }
        )
        with (question_dir / "result.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        entries.append(result)

    try:
        next(result_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(f"{result_path} has more entries than {tsv_path}")

    summary = write_summary(output_dir, entries)
    print(f"\n{output_dir}: {len(entries)} entries, {len({s.image_number for s in samples})} images")
    for category in ("single", "cross"):
        stats = summary[category]
        print(
            f"  {category}: total={stats['total']}, correct={stats['correct']}, "
            f"average_num_tool={stats['average_num_tool']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True, help="Model name used in result filenames")
    parser.add_argument("--hrbench_path", required=True, type=Path, help="HR-Bench directory containing hr_bench_4k.tsv and hr_bench_8k.tsv")
    parser.add_argument("--save_path", required=True, type=Path, help="Directory containing the model result directory")
    parser.add_argument("--mode", choices=("4k", "8k", "both"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.save_path / args.model_name
    modes = ("4k", "8k") if args.mode == "both" else (args.mode,)
    try:
        for mode in modes:
            extract_mode(
                mode=mode,
                model_name=args.model_name,
                hrbench_path=args.hrbench_path,
                result_dir=result_dir,
            )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
