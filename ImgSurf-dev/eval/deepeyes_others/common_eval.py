"""Shared OpenAI-compatible evaluation runner for the additional DeepEyes benchmarks.

The benchmark releases use several annotation formats.  This module intentionally
keeps the dataset-specific entry points small while accepting the common fields
used by the official releases (question/text, answer/label/ground_truth and
image/image_path/image_file).  It also accepts JSON, JSONL, CSV/TSV and Parquet
manifests so that the downloaded files do not have to be converted by hand.

The runner sends the image as a data URL.  This is supported by both vLLM and
SGLang OpenAI-compatible servers and works for Qwen2.5-VL and Qwen3-VL.  For
grounding datasets coordinates are requested in the Qwen3 0--1000 convention;
the judge converts reference boxes to the same convention when image dimensions
are available.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - only needed when local images are used
    Image = None


DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mme_real": {"name": "MME-Real", "task": "vqa", "citation": "mme"},
    "mme_real_cn": {"name": "MME-Real-CN", "task": "vqa", "citation": "mme"},
    "mme_real_lite": {"name": "MME-Real-Lite", "task": "vqa", "citation": "mme"},
    "mathvista_mini": {"name": "MathVista-Mini", "task": "vqa", "citation": "mathvista"},
    "mathverse_mini": {"name": "MathVerse-Mini", "task": "vqa", "citation": "mathverse"},
    "mathverse_vision": {"name": "MathVerse-Vision", "task": "vqa", "citation": "mathverse"},
    "mathvision_mini": {"name": "MathVision-Mini", "task": "vqa", "citation": "mathvision"},
    "wemath_overall": {"name": "WeMath-Overall", "task": "vqa", "citation": "wemath"},
    "dynamath_overall": {"name": "DynaMath-Overall", "task": "vqa", "citation": "dynamath"},
    "logicvista_overall": {"name": "LogicVista-Overall", "task": "vqa", "citation": "logicvista"},
    "ocrbench": {"name": "OCRBench", "task": "vqa", "citation": "ocrbench"},
    "refcoco": {"name": "refCOCO", "task": "grounding", "citation": "refcoco"},
    "refcoco_plus": {"name": "refCOCO+", "task": "grounding", "citation": "refcoco"},
    "refcocog": {"name": "refCOCOg", "task": "grounding", "citation": "refcocog"},
    "reasonseg": {"name": "ReasonSeg", "task": "grounding", "citation": "reasonseg"},
    "pope": {"name": "POPE", "task": "yes_no", "citation": "pope"},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_IMAGE_INDEX_CACHE: Dict[str, Dict[str, str]] = {}
QUESTION_KEYS = (
    "question", "Question", "query", "text", "Text", "prompt", "question_for_eval", "query_wo",
)
ANSWER_KEYS = (
    "answer", "answers", "gt_answer", "ground_truth", "Ground truth", "label", "target", "correct_answer",
)
IMAGE_KEYS = (
    "image", "Image", "image_path", "image_file", "image_name", "imagename", "img", "file_name", "filename",
    "image_base64", "base64_image", "image_data",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _first(record: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _flatten_records(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if not isinstance(value, dict):
        return []
    for key in ("data", "annotations", "questions", "items", "samples", "examples", "test", "testmini", "val", "validation"):
        child = value.get(key)
        if isinstance(child, list):
            return [x for x in child if isinstance(x, dict)]
        if isinstance(child, dict):
            nested = _flatten_records(child)
            if nested:
                return nested
    grouped: List[Dict[str, Any]] = []
    for child in value.values():
        if isinstance(child, list):
            # Do not interpret arbitrary list-valued metadata as samples.
            # ReasonSeg records contain a `shapes` list whose dictionaries have
            # `label`/`image_name`, but the parent record carries the actual
            # question and polygon.  Only flatten lists that look like
            # question/answer records.
            grouped.extend(
                item
                for item in child
                if isinstance(item, dict)
                and any(
                    key in item
                    for key in QUESTION_KEYS
                    + ("instruction", "caption", "answer", "answers", "gt_answer", "ground_truth", "Ground truth", "correct_answer")
                )
            )
    if grouped:
        return grouped
    keyed_records = [
        child for child in value.values()
        if isinstance(child, dict) and any(key in child for key in QUESTION_KEYS + IMAGE_KEYS + ANSWER_KEYS)
    ]
    if keyed_records:
        return keyed_records
    return [value] if any(k in value for k in QUESTION_KEYS + IMAGE_KEYS + ANSWER_KEYS) else []


def _read_table(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jsonlines"}:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            rows.append(value)
                    except json.JSONDecodeError:
                        continue
        return rows
    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                return _flatten_records(json.load(handle))
        except (OSError, json.JSONDecodeError):
            return []
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8", newline="") as handle:
            delimiter = "\t" if suffix == ".tsv" else ","
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix == ".parquet":
        try:
            import pandas as pd

            return pd.read_parquet(path).to_dict(orient="records")
        except Exception:
            try:
                import pyarrow.parquet as pq

                return pq.read_table(path).to_pylist()
            except Exception:
                # Several HF snapshots keep both a JSON manifest and a
                # duplicate Parquet export.  A missing optional Parquet
                # engine must not prevent the JSON copy from being read.
                # load_records() reports a focused dependency error when a
                # dataset has no other usable manifest.
                return []
    return []


def _annotation_files(root: Path) -> Iterator[Path]:
    ignored = {"predictions", "results", "outputs", "logs", "work_dirs", "eval", "label_studio", "tests"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".jsonlines", ".csv", ".tsv", ".parquet"}:
            continue
        if any(part.lower() in ignored or part.startswith(".") for part in path.parts):
            continue
        if path.name.lower() in {"package.json", "dataset_info.json", "state.json"}:
            continue
        yield path


def _annotation_allowed(path: Path, dataset: str) -> bool:
    """Apply filters needed when several official subsets share one folder."""
    lowered = str(path).lower().replace("\\", "/")
    name = path.name.lower()
    # This is an auxiliary OCR manifest contained inside MME-HD-CN.  Its
    # samples are already included in MME_RealWorld_CN.json; reading both
    # manifests would duplicate roughly three thousand questions and, because
    # it lives below the shared mme_real root, would also pollute mme_real.
    if name == "cn_table_ocr.json":
        return False
    if dataset in {"refcoco", "refcoco_plus", "refcocog"}:
        # instances*.json describes COCO itself, not a referring-expression
        # question.  It is consumed by _refcoco_records below.
        if name.startswith("instances"):
            return False
    if dataset == "mme_real_cn":
        return "cn" in name or "/cn/" in lowered or "_cn" in lowered
    if dataset == "mme_real":
        return not ("mme_realworld_cn" in lowered or "_cn" in name or "/cn/" in lowered)
    if dataset == "mme_real_lite":
        return "lite" in lowered or "realworld" in lowered
    if dataset in {"mathvista_mini", "mathvision_mini"} and path.suffix.lower() == ".parquet":
        # These downloaded repositories keep both the full test split and its
        # Mini subset.  The evaluator is explicitly the Mini benchmark.
        return "testmini" in name
    if dataset == "pope":
        # POPE's Full directory repeats the three test subsets as standalone
        # files; use the Hugging Face data/test shards once.
        return not any(part.lower() == "full" for part in path.parts)
    return True


def _refcoco_records(root: Path, dataset: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """Read the official refer pickles when a converted JSON manifest is absent."""
    if dataset not in {"refcoco", "refcoco_plus", "refcocog"}:
        return []
    dataset_folder = {
        "refcoco": "refcoco",
        "refcoco_plus": "refcoco+",
        "refcocog": "refcocog",
    }[dataset]
    dataset_root = root / dataset_folder
    # The shared COCO annotation directory contains 500 MB+ of generic
    # instance/keypoint metadata.  Each RefCOCO release already ships its own
    # compact instances.json next to refs*.p; prefer it so loading one split
    # does not deserialize every COCO annotation file or mix the splits.
    instance_files = sorted(dataset_root.glob("instances*.json"))
    if not instance_files:
        instance_files = sorted(set(root.rglob("instances*.json")) | set(root.rglob("instances.json")))
    if not instance_files:
        return []
    images: Dict[str, Dict[str, Any]] = {}
    annotations: Dict[str, Dict[str, Any]] = {}
    for instance_file in instance_files:
        try:
            instance_data = json.loads(instance_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in instance_data.get("images", []):
            if isinstance(item, dict):
                image_id = str(item.get("id"))
                images[image_id] = {
                    "file_name": item.get("file_name", ""),
                    "width": item.get("width"),
                    "height": item.get("height"),
                }
        for item in instance_data.get("annotations", []):
            if isinstance(item, dict):
                annotation_id = str(item.get("id"))
                annotations[annotation_id] = {"bbox": item.get("bbox")}
    if not images or not annotations:
        return []
    refs_files = sorted(dataset_root.glob("refs*.p"))
    if not refs_files:
        refs_files = sorted(root.rglob("*.p"))
    records: List[Dict[str, Any]] = []
    for refs_file in refs_files:
        refs_lower = str(refs_file).lower().replace("\\", "/")
        if dataset == "refcoco_plus" and not ("refcoco+" in refs_lower or "refcoco_plus" in refs_lower):
            continue
        if dataset == "refcocog" and "refcocog" not in refs_lower:
            continue
        if dataset == "refcoco" and ("refcoco+" in refs_lower or "refcoco_plus" in refs_lower or "refcocog" in refs_lower):
            continue
        try:
            with refs_file.open("rb") as handle:
                refs = pickle.load(handle)
        except Exception:
            continue
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            image = images.get(str(ref.get("image_id")), {})
            annotation = annotations.get(str(ref.get("ann_id")), {})
            sentences = ref.get("sentences", [])
            question = ""
            if isinstance(sentences, list) and sentences:
                sentence = sentences[0]
                question = sentence.get("sent", sentence.get("raw", "")) if isinstance(sentence, dict) else str(sentence)
            record = {
                "id": ref.get("ref_id", ref.get("ann_id", "")),
                "image": image.get("file_name", ""),
                "question": question,
                "answer": "",
                "bbox": annotation.get("bbox"),
                "image_id": ref.get("image_id"),
                "split": ref.get("split", ""),
                "width": image.get("width"),
                "height": image.get("height"),
            }
            normalised = _normalise_record(record, root, refs_file)
            if normalised:
                records.append(normalised)
                if max_samples > 0 and len(records) >= max_samples:
                    return records
    return records


def _image_files(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def _image_index(root: Path) -> Dict[str, str]:
    key = str(root)
    cached = _IMAGE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    index: Dict[str, str] = {}
    for path in _image_files(root):
        # Keep the first sorted path for basename fallback.  Dataset-specific
        # relative paths are resolved before this fallback, so repeated names
        # such as DynaMath's trial*/image/image1.png remain unambiguous.
        index.setdefault(path.name, str(path))
    _IMAGE_INDEX_CACHE[key] = index
    return index


def _image_dimensions(path: str) -> Optional[Tuple[int, int]]:
    """Read common image dimensions without requiring Pillow."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                import struct

                return struct.unpack(">II", header[16:24])
            if header.startswith(b"GIF") and len(header) >= 10:
                import struct

                return struct.unpack("<HH", header[6:10])
            if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                if header[12:16] == b"VP8X" and len(header) >= 30:
                    width = 1 + int.from_bytes(header[24:27], "little")
                    height = 1 + int.from_bytes(header[27:30], "little")
                    return width, height
            if header[:2] == b"\xff\xd8":
                handle.seek(2)
                while True:
                    marker_prefix = handle.read(1)
                    if not marker_prefix:
                        break
                    if marker_prefix != b"\xff":
                        continue
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if not marker or marker in {b"\xd8", b"\xd9"}:
                        continue
                    size_bytes = handle.read(2)
                    if len(size_bytes) != 2:
                        break
                    size = int.from_bytes(size_bytes, "big")
                    if size < 2:
                        break
                    if marker[0] in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                        dimensions = handle.read(5)
                        if len(dimensions) == 5:
                            height = int.from_bytes(dimensions[1:3], "big")
                            width = int.from_bytes(dimensions[3:5], "big")
                            return width, height
                    handle.seek(size - 2, 1)
    except (OSError, ValueError):
        pass
    return None


def _record_image(record: Dict[str, Any], root: Path, source: Optional[Path] = None) -> Tuple[Optional[str], Optional[bytes]]:
    value = _first(record, IMAGE_KEYS)
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), bytes):
            return None, value["bytes"]
        if isinstance(value.get("bytes"), str):
            try:
                return None, base64.b64decode(value["bytes"])
            except Exception:
                pass
        value = value.get("path") or value.get("filename") or value.get("file_name")
    if isinstance(value, list) and value:
        value = value[0]
    if not value:
        return None, None
    if isinstance(value, (bytes, bytearray)):
        return None, bytes(value)
    if isinstance(value, str) and value.startswith("data:image/") and "," in value:
        try:
            return None, base64.b64decode(value.split(",", 1)[1])
        except Exception:
            pass
    if isinstance(value, str) and len(value) > 512 and not Path(value).suffix:
        # MME-RealWorld JSON stores some images as raw base64 strings.
        try:
            decoded = base64.b64decode(value, validate=True)
            if decoded[:2] == b"\xff\xd8" or decoded[:8] == b"\x89PNG\r\n\x1a\n":
                return None, decoded
        except Exception:
            pass
    candidate = Path(str(value))
    if candidate.is_absolute() and candidate.is_file():
        return str(candidate), None
    # HF dataset archives commonly add an `images/` or `data/` prefix while
    # manifests keep the path relative to the logical dataset root.  Try these
    # prefixes before falling back to a basename search (which can be ambiguous
    # for MathVerse's versioned image directories).
    direct_candidates = []
    if source is not None:
        direct_candidates.append(source.parent / candidate)
    direct_candidates.extend([
        root / candidate,
        root / "images" / candidate,
        root / "data" / candidate,
        root / "data" / "images" / candidate,
        root / "train2014" / candidate,
        root / "val2014" / candidate,
    ])
    for direct in direct_candidates:
        if direct.is_file():
            return str(direct), None
    indexed = _image_index(root).get(candidate.name)
    if indexed:
        return indexed, None
    # Do not keep an unresolved manifest value as if it were a valid local
    # image.  That would create apparently loadable rows that later send a
    # text-only request (for example MathVista's source metadata uses values
    # such as "plot" in its `image` field).
    return None, None


def _normalise_record(record: Dict[str, Any], root: Path, source: Path) -> Optional[Dict[str, Any]]:
    image_path, image_bytes = _record_image(record, root, source)
    if not image_path and image_bytes is None:
        for extension in IMAGE_EXTENSIONS:
            sibling = source.with_suffix(extension)
            if sibling.is_file():
                image_path = str(sibling)
                break
    question = _first(record, QUESTION_KEYS, "")
    if isinstance(question, (list, tuple)):
        question = question[0] if question else ""
    answer = _first(record, ANSWER_KEYS, "")
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else ""
    if isinstance(answer, dict):
        answer = _first(answer, ("text", "value", "label", "answer"), json.dumps(answer, ensure_ascii=False))
    if not image_path and image_bytes is None:
        return None
    if not question:
        question = str(record.get("instruction", record.get("caption", "")))
    if not str(question).strip():
        return None
    row = dict(record)
    row.update(
        {
            "question": str(question),
            "answer": answer,
            "image_path": image_path,
            "_image_bytes": image_bytes,
            "_source": str(source),
        }
    )
    if image_path and Image is not None:
        try:
            with Image.open(image_path) as image:
                row.setdefault("width", image.width)
                row.setdefault("height", image.height)
        except Exception:
            pass
    if image_path and (not row.get("width") or not row.get("height")):
        dimensions = _image_dimensions(image_path)
        if dimensions:
            row.setdefault("width", dimensions[0])
            row.setdefault("height", dimensions[1])
    return row


def load_records(dataset_path: str, max_samples: int = 0, dataset: str = "") -> List[Dict[str, Any]]:
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    # RefCOCO is fully described by the compact per-split instances.json and
    # refs*.p files.  Do not walk the shared 120k-image COCO tree a second
    # time looking for unrelated annotation copies.
    annotation_paths = [] if dataset in {"refcoco", "refcoco_plus", "refcocog"} else list(_annotation_files(root))
    records: List[Dict[str, Any]] = _refcoco_records(root, dataset, max_samples=max_samples)
    if dataset in {"refcoco", "refcoco_plus", "refcocog"}:
        return records[:max_samples] if max_samples > 0 else records
    target_reached = max_samples > 0 and len(records) >= max_samples
    if target_reached:
        return records[:max_samples]
    non_parquet_records = False
    for path in annotation_paths:
        # Official HF snapshots often include both JSON and Parquet exports of
        # the same split.  Prefer a usable JSON/JSONL export and avoid adding a
        # second copy when a Parquet engine is installed.  Parquet remains the
        # fallback for datasets that have no usable text manifest.
        if path.suffix.lower() == ".parquet" and non_parquet_records:
            continue
        if not _annotation_allowed(path, dataset):
            continue
        before = len(records)
        for record in _read_table(path):
            normalised = _normalise_record(record, root, path)
            if normalised:
                records.append(normalised)
        if path.suffix.lower() != ".parquet" and len(records) > before:
            non_parquet_records = True
    # Some releases store one JSON file next to every image.  The loop above
    # handles those; this fallback only creates records when no manifest was found.
    if not records:
        parquet_files = [path for path in annotation_paths if path.suffix.lower() == ".parquet"]
        if parquet_files:
            raise RuntimeError(
                f"No readable records found under {root}. This dataset only has Parquet manifests; "
                "install pandas or pyarrow before evaluation."
            )
        if annotation_paths:
            raise RuntimeError(f"No usable image/annotation records found under {root}")
        for image in _image_files(root):
            records.append(
                {
                    "question": "Describe the image and answer the benchmark question.",
                    "answer": "",
                    "image_path": str(image),
                    "_image_bytes": None,
                    "_source": str(image),
                }
            )
    # Avoid duplicated rows when both a raw JSON and converted Parquet copy are present.
    unique: Dict[Tuple[str, str, str, str], Tuple[str, Dict[str, Any]]] = {}
    deduplicated: List[Dict[str, Any]] = []
    for row in records:
        # MathVerse deliberately contains several problem versions that can
        # share the same rendered image, question and answer.  Keep those
        # variants distinct while still collapsing duplicate JSON/Parquet
        # exports of the same sample.
        variant = row.get("problem_version", row.get("version", ""))
        key = (
            str(row.get("image_path", "")),
            str(row.get("question", "")),
            str(row.get("answer", "")),
            str(variant),
        )
        source = str(row.get("_source", ""))
        previous = unique.get(key)
        if previous is None:
            unique[key] = (source, row)
            deduplicated.append(row)
        elif previous[0] == source:
            # Repeated questions in one official manifest are valid samples
            # (MME-Lite contains several repeated IDs with distinct rows).
            deduplicated.append(row)
        # A matching row from a second export is the duplicate copy to drop.
    result = deduplicated
    return result[:max_samples] if max_samples > 0 else result


def _matches_split(row: Dict[str, Any], split: Optional[str]) -> bool:
    if not split:
        return True
    wanted = split.lower()
    values: List[str] = []
    for key in ("split", "subset", "partition", "set"):
        if row.get(key) is not None:
            values.append(str(row[key]))
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("split", "subset", "partition", "set"):
            if metadata.get(key) is not None:
                values.append(str(metadata[key]))
    values.append(str(row.get("_source", "")))
    return any(wanted == value.lower() or wanted in value.lower() for value in values)


def _matches_problem_version(row: Dict[str, Any], problem_version: Optional[str]) -> bool:
    if not problem_version:
        return True
    wanted = problem_version.strip().lower()
    values = [str(row.get("problem_version", "")), str(row.get("version", ""))]
    return any(wanted == value.strip().lower() or wanted in value.lower() for value in values if value)


def _mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp", ".gif": "gif", ".bmp": "bmp"}.get(ext, "png")


def image_data_url(row: Dict[str, Any]) -> Optional[str]:
    raw = row.get("_image_bytes")
    if raw:
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    image_path = row.get("image_path")
    if not image_path:
        return None
    path = Path(str(image_path))
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return f"data:image/{_mime(str(path))};base64," + base64.b64encode(handle.read()).decode("ascii")


def _strip_option_marker(value: Any) -> str:
    return re.sub(r"^\s*\(?[A-Z]\)?\s*[:.)]\s*", "", str(value))


def _options(record: Dict[str, Any]) -> str:
    options = record.get("options", record.get("choices"))
    if options in (None, ""):
        options = record.get("Answer choices", record.get("option"))
    if isinstance(options, dict):
        return "\n".join(f"{key}. {value}" for key, value in options.items())
    if isinstance(options, (list, tuple)):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{letters[i]}. {_strip_option_marker(value)}" for i, value in enumerate(options))
    if isinstance(options, str):
        return options
    # A few MCQ releases store choices as A/B/C/D columns.
    pairs = [(key, record[key]) for key in "ABCDEF" if key in record]
    return "\n".join(f"{key}. {value}" for key, value in pairs)


def _is_grounding(dataset: str, record: Dict[str, Any]) -> bool:
    return DATASET_CONFIGS.get(dataset, {}).get("task") == "grounding" or any(
        key in record for key in ("bbox", "bbox_2d", "box", "grounding", "shapes", "polygons")
    )


def prompt_for(dataset: str, version: str, record: Dict[str, Any], qwen_ver: int) -> str:
    question = record.get("question", "")
    options = _options(record)
    option_text = f"\nOptions:\n{options}" if options else ""
    if _is_grounding(dataset, record):
        output = "Return one bounding box in <answer>[x1, y1, x2, y2]</answer>. Use coordinates normalized to 0-1000 in the image currently shown."
    elif DATASET_CONFIGS.get(dataset, {}).get("task") == "yes_no":
        output = "Answer only yes or no inside <answer>...</answer>."
    else:
        output = "Give the final answer inside <answer>...</answer>. Keep it concise and preserve an option letter when the question is multiple choice."
    if version == "v0":
        reasoning = "Answer directly; do not emit a think block."
    elif version == "v2":
        reasoning = "Reason briefly, then emit <answer>...</answer>."
    else:
        reasoning = "Think step by step in <think>...</think>, then emit exactly one <answer>...</answer>."
    qwen_note = "Qwen3-VL" if int(qwen_ver) == 3 else "Qwen2.5-VL"
    return (
        f"You are evaluating {DATASET_CONFIGS.get(dataset, {}).get('name', dataset)} with {qwen_note}.\n"
        f"Question: {question}{option_text}\n{reasoning}\n{output}"
    )


def _client(api_url: str, api_key: str):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the openai package to use the OpenAI-compatible evaluator") from exc
    return OpenAI(api_key=api_key, base_url=api_url)


def _model_name(client: Any, requested: Optional[str]) -> str:
    if requested:
        return requested
    response = client.models.list()
    if not response.data:
        raise RuntimeError("The API returned no models; pass --eval_model_name explicitly")
    return response.data[0].id


def _call(client: Any, model: str, messages: List[Dict[str, Any]], max_tokens: int, retries: int = 3) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return (result.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover - endpoint-dependent
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"VLM request failed after {retries} attempts: {last_error}")


def _answer_from_response(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text or "", flags=re.I | re.S)
    return match.group(1).strip() if match else (text or "").strip()


def _eval_one(client: Any, model: str, dataset: str, version: str, row: Dict[str, Any], qwen_ver: int, max_tokens: int) -> Dict[str, Any]:
    data_url = image_data_url(row)
    content: List[Dict[str, Any]] = []
    if data_url:
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    content.append({"type": "text", "text": prompt_for(dataset, version, row, qwen_ver)})
    messages = [
        {"role": "system", "content": "You are a careful visual question answering assistant."},
        {"role": "user", "content": content},
    ]
    try:
        raw = _call(client, model, messages, max_tokens)
        status = "success"
        error = ""
    except Exception as exc:
        raw = ""
        status = "error"
        error = repr(exc)
    return {
        "id": row.get("id", row.get("question_id", row.get("pid", ""))),
        "image": row.get("image_path"),
        "question": row.get("question", ""),
        "answer": _jsonable(row.get("answer", "")),
        "pred_ans": _answer_from_response(raw),
        "pred_output": raw,
        "status": status,
        "error": error,
        "qwen_ver": int(qwen_ver),
        "version": version,
        "record": _jsonable({k: v for k, v in row.items() if not k.startswith("_")}),
    }


def _add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_path", "--data_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--model_name", default="qwen-vl")
    parser.add_argument("--eval_model_name", default=None)
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--api_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--split", default=None, help="Optional official split name, for example testmini or test")
    parser.add_argument("--problem_version", default=None, help="MathVerse version, for example Vision Only")
    parser.add_argument("--qwen_ver", type=int, choices=(2, 3), default=3)
    parser.add_argument("--max_tokens", type=int, default=4096)


def main_eval(dataset: str, version: str, argv: Optional[Sequence[str]] = None, default_problem_version: Optional[str] = None) -> Path:
    parser = argparse.ArgumentParser(description=f"Evaluate {DATASET_CONFIGS.get(dataset, {}).get('name', dataset)} ({version})")
    _add_common_eval_args(parser)
    args = parser.parse_args(argv)
    problem_version = args.problem_version or default_problem_version
    # Filtering must happen before truncation.  In particular, the first rows
    # of MathVerse are Text Dominant, so applying --max_samples while loading
    # would leave the mathverse_vision logical split empty.
    load_limit = args.max_samples if not args.split and not problem_version else 0
    rows = load_records(args.dataset_path, load_limit, dataset)
    if args.split:
        rows = [row for row in rows if _matches_split(row, args.split)]
    if problem_version:
        rows = [row for row in rows if _matches_problem_version(row, problem_version)]
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    if not rows:
        raise RuntimeError(f"No image/annotation records found under {args.dataset_path}")
    client = _client(args.api_url, args.api_key)
    model = _model_name(client, args.eval_model_name)
    output_dir = Path(args.save_path).expanduser() / args.model_name / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"predictions_{version}.jsonl"
    results: List[Dict[str, Any]] = []
    workers = max(1, min(int(args.num_workers), len(rows)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_eval_one, client, model, dataset, version, row, args.qwen_ver, args.max_tokens) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    # Keep deterministic order for judge scripts and comparison between versions.
    results.sort(key=lambda x: str(x.get("id", "")) + str(x.get("image", "")))
    with output_file.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    metadata = {"dataset": dataset, "version": version, "model": model, "qwen_ver": args.qwen_ver, "count": len(results)}
    (output_dir / f"metadata_{version}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} predictions to {output_file}")
    return output_file


def _clean_answer(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"<answer>|</answer>|\\boxed\s*\{?", "", text, flags=re.I)
    text = text.strip().strip("`*_ {}").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ",", text)
    return text


def _answer_options(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] if value not in (None, "") else []


def _choice_set(value: Any) -> Optional[frozenset[str]]:
    parts = [part for part in re.split(r"[\s,]+", _clean_answer(value)) if part]
    if len(parts) > 1 and all(re.fullmatch(r"[a-z]", part) for part in parts):
        return frozenset(parts)
    return None


def _exact_match(pred: Any, truth: Any, record: Dict[str, Any], dataset: str) -> bool:
    p = _clean_answer(pred)
    truths = [_clean_answer(v) for v in _answer_options(truth)]
    if not p or not truths:
        return False
    if DATASET_CONFIGS.get(dataset, {}).get("task") == "yes_no":
        p = "yes" if re.search(r"\byes\b", p) and not re.search(r"\bno\b", p) else "no" if re.search(r"\bno\b", p) else p
        truths = ["yes" if "yes" in t else "no" if "no" in t else t for t in truths]
    predicted_choices = _choice_set(p)
    if predicted_choices:
        for truth_value in truths:
            if predicted_choices == _choice_set(truth_value):
                return True
    options = record.get("options", record.get("choices"))
    if options in (None, ""):
        options = record.get("Answer choices", record.get("option"))
    if isinstance(options, str):
        parsed = re.findall(r"(?:^|[;\n])\s*\(?([A-Z])\)?\s*[:.)]\s*([^;\n]+)", options, flags=re.I)
        if parsed:
            options = [value.strip() for _, value in parsed]
    if isinstance(options, (list, tuple)):
        letters = "abcdefghijklmnopqrstuvwxyz"
        for idx, option in enumerate(options):
            option_clean = _clean_answer(_strip_option_marker(option))
            if p == letters[idx] or p.startswith(letters[idx] + ".") or p == option_clean or option_clean and option_clean in p:
                if any(
                    t == option_clean
                    or t == letters[idx]
                    or t.startswith(letters[idx] + ".")
                    or (t.isdigit() and int(t) in {idx, idx + 1})
                    for t in truths
                ):
                    return True
    for truth_value in truths:
        try:
            if abs(float(p.replace(",", "")) - float(truth_value.replace(",", ""))) < 1e-5:
                return True
        except (TypeError, ValueError):
            pass
    return any(p == t or (len(p) > 2 and (p.startswith(t) or t.startswith(p))) for t in truths)


def _numbers(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(x) for x in list(value)[:4]]
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        for key in ("bbox_2d", "bbox", "box", "xyxy"):
            found = _numbers(value.get(key))
            if found:
                return found
    if isinstance(value, str):
        match = re.search(r"\[?\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*\]?", value)
        if match:
            return [float(match.group(i)) for i in range(1, 5)]
    return None


def _ground_truth_bbox(record: Dict[str, Any], dataset: str = "") -> Optional[List[float]]:
    for key in ("bbox_2d", "bbox", "box", "grounding", "ground_truth_bbox"):
        found = _numbers(record.get(key))
        if found:
            # COCO annotations use [x, y, width, height], while model answers
            # and bbox_2d fields use [x1, y1, x2, y2].
            if key == "bbox" and dataset in {"refcoco", "refcoco_plus", "refcocog"}:
                found = [found[0], found[1], found[0] + found[2], found[1] + found[3]]
            return found
    shapes = record.get("shapes", record.get("polygons"))
    points: List[Tuple[float, float]] = []
    if isinstance(shapes, list):
        for shape in shapes:
            if isinstance(shape, dict) and str(shape.get("label", "target")).lower() not in {"ignore", "background"}:
                candidate = shape.get("points", shape.get("polygon"))
                if isinstance(candidate, list):
                    for point in candidate:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            try:
                                points.append((float(point[0]), float(point[1])))
                            except (TypeError, ValueError):
                                pass
    if points:
        return [min(x for x, _ in points), min(y for _, y in points), max(x for x, _ in points), max(y for _, y in points)]
    return None


def _normalise_box(box: List[float], width: Optional[float], height: Optional[float], force_pixels: bool = False) -> List[float]:
    # Reference annotations are often pixels.  Prediction coordinates are in
    # Qwen's 0--1000 convention.  Convert only when dimensions make this safe.
    if width and height and (force_pixels or max(box) > 1000 or max(box) > max(width, height)):
        return [box[0] / width * 1000, box[1] / height * 1000, box[2] / width * 1000, box[3] / height * 1000]
    return box


def _iou(a: List[float], b: List[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def _judge_prompt(item: Dict[str, Any]) -> str:
    return (
        "Decide whether the model answer is correct. Return only 1 for correct or 0 for incorrect.\n"
        f"Question: {item.get('question', '')}\n"
        f"Reference answer: {item.get('answer', '')}\n"
        f"Model answer: {item.get('pred_ans', '')}"
    )


def _judge_one(client: Any, model: str, item: Dict[str, Any], dataset: str, use_vlm: bool) -> Dict[str, Any]:
    record = item.get("record", {}) if isinstance(item.get("record"), dict) else {}
    truth = item.get("answer", record.get("answer", ""))
    pred = item.get("pred_ans", "")
    task = DATASET_CONFIGS.get(dataset, {}).get("task")
    iou: Optional[float] = None
    correct = _exact_match(pred, truth, record, dataset)
    if task == "grounding":
        pred_box = _numbers(pred)
        gt_box = _ground_truth_bbox(record, dataset)
        if pred_box and gt_box:
            width = record.get("width", record.get("img_width"))
            height = record.get("height", record.get("img_height"))
            gt_box = _normalise_box(
                gt_box,
                float(width) if width else None,
                float(height) if height else None,
                force_pixels=dataset in {"refcoco", "refcoco_plus", "refcocog", "reasonseg"},
            )
            iou = _iou(pred_box, gt_box)
            correct = iou >= 0.5
    if not correct and use_vlm and client is not None and task != "grounding":
        try:
            response = _call(
                client,
                model,
                [{"role": "system", "content": "You are a strict benchmark judge."}, {"role": "user", "content": _judge_prompt(item)}],
                64,
            )
            correct = bool(re.search(r"\b1\b", response)) and not bool(re.search(r"\b0\b", response))
            item["judge_raw"] = response
        except Exception as exc:
            item["judge_error"] = repr(exc)
    item["correct"] = bool(correct)
    item["score"] = 1.0 if correct else 0.0
    if iou is not None:
        item["bbox_iou"] = iou
    return item


def _add_common_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_path", "--data_path", default=None)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--prediction_file", default=None)
    parser.add_argument("--model_name", default="qwen-vl")
    parser.add_argument("--eval_model_name", default=None)
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--api_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_vlm_judge", action="store_true")


def main_judge(dataset: str, argv: Optional[Sequence[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description=f"Judge {DATASET_CONFIGS.get(dataset, {}).get('name', dataset)} predictions")
    _add_common_judge_args(parser)
    args = parser.parse_args(argv)
    output_dir = Path(args.save_path).expanduser() / args.model_name / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = Path(args.prediction_file).expanduser() if args.prediction_file else output_dir / "predictions_v4.jsonl"
    if not prediction.exists() and not args.prediction_file:
        candidates = sorted(output_dir.glob("predictions_*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No prediction JSONL found under {output_dir}")
        prediction = candidates[-1]
    with prediction.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    client = _client(args.api_url, args.api_key) if args.use_vlm_judge else None
    model = _model_name(client, args.eval_model_name) if client is not None else ""
    results = [_judge_one(client, model, item, dataset, args.use_vlm_judge) for item in items]
    score = sum(float(item.get("score", 0.0)) for item in results) / max(len(results), 1)
    output_file = prediction.with_name(prediction.stem + "_judge.jsonl")
    with output_file.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {"dataset": dataset, "prediction_file": str(prediction), "count": len(results), "accuracy": score}
    if dataset == "pope":
        tp = fp = fn = tn = 0
        for item in results:
            truth_text = _clean_answer(item.get("answer", ""))
            pred_text = _clean_answer(item.get("pred_ans", ""))
            truth = "yes" if re.search(r"\byes\b", truth_text) and not re.search(r"\bno\b", truth_text) else "no"
            pred = "yes" if re.search(r"\byes\b", pred_text) and not re.search(r"\bno\b", pred_text) else "no"
            if truth == "yes" and pred == "yes":
                tp += 1
            elif truth == "no" and pred == "yes":
                fp += 1
            elif truth == "yes" and pred == "no":
                fn += 1
            else:
                tn += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        summary.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "yes_ratio": (tp + fp) / max(len(results), 1),
            }
        )
    summary_file = output_dir / f"judge_result_{prediction.stem.replace('predictions_', '')}.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary_file

