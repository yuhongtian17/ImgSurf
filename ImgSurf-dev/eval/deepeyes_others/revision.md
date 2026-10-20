# Revision notes

## Shared benchmark layout

- `DATA_ROOT` now defaults to `../benchmarks`, matching the server layout
  `/home/dataset-assist-0/workspace/deepeyes271/benchmarks`.
- `mme_real_cn` is a logical evaluation name that reads
  `MME_RealWorld_CN.json` from `mme_real`; no separate CN directory is used.
- `mathverse_vision` is a logical evaluation name that reads the shared
  `mathverse_mini` directory and filters `problem_version == "Vision Only"`.
- `refcoco`, `refcoco_plus`, `refcocog` all read the shared `refcoco` directory,
  which also contains the extracted MSCOCO2014 images and annotations.
- RefCOCO pickle files are selected by dataset name and COCO `instances*.json`
  files are not treated as VQA manifests.

## Archive inspection

- Added `check_benchmarks.sh`. Its default mode is read-only and reports
  `NEEDS_EXTRACT` for ZIP/TAR/7z payloads that are not present beside the archive.
- The script detects MME `.part_*` shards and, with `EXTRACT=1`, concatenates and
  extracts them. It can also extract ordinary ZIP/TAR/7z archives in place.
- The README documents the exact server commands and explains that the current
  archive status must be obtained on the GPU server because benchmark files are
  outside this checkout.

After the later extraction pass, the supplied benchmark tree contains the
payload of every downloaded ZIP/TAR/7z archive. The original archive files are
kept for reproducibility. The nested
`mme_real/MME-HD-CN/ch_monitoring_images.zip` uses the existing
`MME-HD-CN/monitoring_images` directory; `check_benchmarks.sh` recognizes this
layout and does not report its 349 images as missing.
The RefCOCO tree also retains multipart `train2014_rar`/`val2014_rar` source
archives, but the corresponding COCO image directories are already complete
(`82783` train images and `40504` validation images), so those RARs are not a
runtime extraction dependency.

## Evaluation changes

- `eval.sh` maps the three shared physical roots to their logical dataset
  evaluators while keeping separate logical prediction output directories.
- `download.sh` follows the shared layout and extracts the RefCOCO/COCO archives
  into the shared `refcoco` root.
- `common_eval.py` filters MME CN/non-CN manifests and reads shared RefCOCO
  annotations without duplicating records from COCO instance manifests.
- The loader now accepts the capitalized MME fields (`Image`, `Text`, `Ground
  truth`, `Answer choices`), LogicVista's `imagename`, WeMath's singular
  `option`, and archive-added `images/`/`data/` path prefixes.
- Repository test fixtures under LogicVista's `label_studio`/`tests` trees and
  Hugging Face cache metadata are ignored so they cannot become evaluation
  samples; keyed JSON dictionaries such as LogicVista's `data/dataset.json`
  are expanded into records.
- README now calls out `pyarrow`, required for the Parquet-only HF releases
  such as OCRBench, POPE, MathVision and MathVista.
- Exact judging normalizes comma-separated multi-choice answers, which covers
  LogicVista records such as `A, C`.
- Read-flow validation now skips duplicate Parquet exports when a JSON manifest
  is usable, while retaining Parquet as the required fallback for Parquet-only
  datasets. Missing optional engines now produce a focused dependency error.
- Relative image paths are resolved relative to their manifest (fixing the
  repeated `trial*/image/image1.png` names in DynaMath), unresolved image
  metadata is discarded, and empty-question fixtures are skipped.
- RefCOCO loading uses only the selected split's compact `instances.json` and
  `refs*.p`, honors `--max_samples` during loading, and avoids walking the full
  COCO image tree before evaluation.
- MME's auxiliary `MME-HD-CN/cn_table_ocr.json` is excluded because its rows
  are already covered by `MME_RealWorld_CN.json`; this prevents CN leakage into
  MME-Real and duplicate CN questions.
- Parquet discovery is restricted to MathVista/MathVision `testmini` shards and
  POPE's `data/test-*` shards; full-test and duplicate POPE exports are ignored.
- Grounding records now retain COCO image dimensions and use a small PNG/JPEG/
  GIF/WebP header fallback when Pillow is unavailable, so RefCOCO and ReasonSeg
  boxes still use the required 0--1000 normalization.
