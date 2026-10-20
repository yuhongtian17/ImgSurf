# Additional DeepEyes evaluations

`deepeyes_others` is an OpenAI-compatible evaluation harness for the benchmarks
used by ImgSurf outside the DeepEyes training set.  It supports a local Qwen2.5-VL
or Qwen3-VL served by vLLM/SGLang and keeps the benchmark-specific command lines
small: every dataset directory contains `eval_<dataset>_v0.py`,
`eval_<dataset>_v2.py`, `eval_<dataset>_v4.py` and
`judge_result_<dataset>.py`.  The shared implementation is in
`common_eval.py`.

The citation keys below are the keys in
[`TIP_ImgSurf/myrefs.bib`](../../../TIP_ImgSurf/myrefs.bib); that BibTeX file
remains the source of the full references.

| Dataset | Logical eval name | Physical data path under `DATA_ROOT` | Citation key | Official data/repository |
| --- | --- | --- | --- | --- |
| MME-Real | `mme_real` | `mme_real` | `mme` | [MME-RealWorld](https://github.com/MME-Benchmarks/MME-RealWorld), [HF release](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld) |
| MME-Real-CN | `mme_real_cn` | `mme_real` (`MME_RealWorld_CN.json`) | `mme` | same release, `MME_RealWorld_CN.json` |
| MME-Real-Lite | `mme_real_lite` | `mme_real_lite` | `mme` | [official lite release](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite), 50 samples per task |
| MathVista-Mini | `mathvista_mini` | `mathvista_mini` | `mathvista` | [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista), `testmini` split |
| MathVerse-Mini | `mathverse_mini` | `mathverse_mini` | `mathverse` | [AI4Math/MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse), `testmini` split |
| MathVerse-Vision | `mathverse_vision` | `mathverse_mini` (`problem_version == "Vision Only"`) | `mathverse` | same release, Vision/vision-only testmini split |
| MathVision-Mini | `mathvision_mini` | `mathvision_mini` | `mathvision` | [MathLLMs/MathVision](https://huggingface.co/datasets/MathLLMs/MathVision), `testmini` split |
| WeMath-Overall | `wemath_overall` | `wemath_overall` | `wemath` | [We-Math/We-Math](https://huggingface.co/datasets/We-Math/We-Math) |
| DynaMath-Overall | `dynamath_overall` | `dynamath_overall` | `dynamath` | [DynaMath/DynaMath_Sample](https://huggingface.co/datasets/DynaMath/DynaMath_Sample) and [official generator](https://github.com/DynaMath/DynaMath) |
| LogicVista-Overall | `logicvista_overall` | `logicvista_overall` | `logicvista` | [Yijia-Xiao/LogicVista](https://github.com/Yijia-Xiao/LogicVista) |
| OCRBench | `ocrbench` | `ocrbench` | `ocrbench` | [OCRBench](https://github.com/qywh2023/OCRbench), [HF mirror](https://huggingface.co/datasets/echo840/OCRBench) |
| refCOCO | `refcoco` | `refcoco` (shared COCO images and annotations) | `refcoco` | [official refer repository](https://github.com/lichengunc/refer) |
| refCOCO+ | `refcoco_plus` | `refcoco` (shared COCO images and annotations) | `refcoco` | same release |
| refCOCOg | `refcocog` | `refcoco` (shared COCO images and annotations) | `refcocog` | same release |
| ReasonSeg | `reasonseg` | `reasonseg` | `reasonseg` | [LISA](https://github.com/dvlab-research/LISA) release |
| POPE | `pope` | `pope` | `pope` | [official POPE repository](https://github.com/RUCAIBox/POPE), [HF format](https://huggingface.co/datasets/lmms-lab-encoder/POPE) |

## Data preparation

Run the download helper from a data volume.  It uses the official Hugging Face
repositories where available, clones LogicVista/LISA, and downloads the archived
RefCOCO annotations.  MME-RealWorld is large; use `hf download --include` if a
full image domain is unnecessary.

The expected server layout is:

```text
/home/dataset-assist-0/workspace/deepeyes271/benchmarks/
├── mme_real/              # also contains MME_RealWorld_CN.json
├── mme_real_lite/
├── mathvista_mini/
├── mathverse_mini/        # also used by the mathverse_vision logical eval
├── mathvision_mini/
├── wemath_overall/  dynamath_overall/  logicvista_overall/  ocrbench/
├── refcoco/                # refCOCO, refCOCO+, refCOCOg and MSCOCO2014
├── reasonseg/  pope/
└── ...                     # downloaded archives before extraction
```

Run the downloader from the checkout, or set `DATA_ROOT` to another data
volume:

```bash
cd /home/dataset-assist-0/workspace/deepeyes271/sCoT-main/eval/deepeyes_others
DATA_ROOT=/home/dataset-assist-0/workspace/deepeyes271/benchmarks bash download.sh
```

The helper cannot automate the ReasonSeg Google Drive confirmation or the COCO
image license download.  After downloading those files, put `val.zip`/`test.zip`
under `$DATA_ROOT/reasonseg` and extract them; put COCO `train2014` images under
`$DATA_ROOT/refcoco` so the RefCOCO annotation paths resolve.  The RefCOCO
evaluators select the correct pickle files from this shared directory, and the
MME loader selects the CN JSON only for the `mme_real_cn` logical eval.  No
`mme_real_cn` or `mathverse_vision` physical directory is required.

For MathVista/MathVerse/MathVision, keep the official `testmini` files in the
corresponding directory.  If the complete HF repository was downloaded, pass
`--split testmini` to an eval script or point `--dataset_path` at a directory
containing only that split.  The MathVerse Vision-only entry points automatically
select `problem_version == "Vision Only"` from `$DATA_ROOT/mathverse_mini`.

### Check and extract downloaded archives

Run this read-only check on the GPU server to identify archives whose contents
are still unavailable to the evaluator:

```bash
cd /home/dataset-assist-0/workspace/deepeyes271/sCoT-main
DATA_ROOT=/home/dataset-assist-0/workspace/deepeyes271/benchmarks \
  bash eval/deepeyes_others/check_benchmarks.sh
```

The command prints `NEEDS_EXTRACT` for ZIP/TAR/7z files with any payload entry
absent, and reports MME `.part_*` shards that must be concatenated.  DynaMath's
`10VariationSamples.zip` contains a nested `.7z` archive; rerun the check after
the outer ZIP is extracted so the nested archive is processed.  After
checking available disk space, `EXTRACT=1` performs the extraction in place:

```bash
DATA_ROOT=/home/dataset-assist-0/workspace/deepeyes271/benchmarks EXTRACT=1 \
  bash eval/deepeyes_others/check_benchmarks.sh
```

Extraction of `.7z` files requires `7z` or `7zz` on the server.  The script
does not delete the original archives.

Run the scan against the actual data volume because benchmark files are kept
outside the code checkout.  It covers archives downloaded from Hugging Face as
well as the ReasonSeg/COCO archives; the output is the exact current status for
the selected `DATA_ROOT`.

Archive files may remain after a successful extraction; their presence alone
does not mean that data is missing.  In the current server snapshot, the
downloaded ZIP/TAR/7z payloads are present.  The nested
`mme_real/MME-HD-CN/ch_monitoring_images.zip` is stored under the existing
`MME-HD-CN/monitoring_images` directory, and the checker accepts this documented
directory rename.  Use the command above against the real `DATA_ROOT` whenever
the benchmark tree changes.

The tree also contains multipart `refcoco/train2014_rar/*.rar` and
`refcoco/val2014_rar/*.rar` source archives.  They are alternate COCO downloads;
the corresponding `refcoco/train2014` and `refcoco/val2014` directories already
contain all images required by the evaluator, so these RAR files do not add an
unextracted runtime dependency.  The checker intentionally focuses on the ZIP,
TAR and 7z formats used by the evaluation loaders.

The loader accepts JSON, JSONL, CSV, TSV and Parquet.  Install `pyarrow` on the
evaluation environment because several Hugging Face releases are Parquet-only:

```bash
pip install -U pandas pyarrow
```

When a dataset contains both JSON and Parquet exports, the loader uses the
usable JSON manifest and skips the duplicate Parquet copy.  MathVision,
OCRBench and POPE currently have only Parquet manifests in the downloaded tree,
so their evaluation stops with an explicit dependency message until `pyarrow`
or `pandas` is installed.
The Mini evaluators select only MathVista/MathVision `testmini` Parquet shards;
POPE selects the `data/test-*` shards and ignores its duplicate `Full/` copies.

Each record should expose
an image field (`image`, `Image`, `image_path`, `image_file`, `image_name` or
`imagename`), a question field (`question`, `Question`, `query` or `text`) and
an answer field (`answer`, `Ground truth`, `label`, `ground_truth` or
`gt_answer`).  `options`/`choices` are optional; MME's `Answer choices` and
WeMath's singular `option` are also supported.  RefCOCO and ReasonSeg records
may additionally contain `bbox`, `bbox_2d`, `box` or polygon `shapes`; these are
scored with IoU >= 0.5.  POPE records are normalized to a yes/no answer.  The
original records are retained in each prediction JSONL so that a dataset-
specific post-processing step can be added without rerunning the VLM.

## Start a Qwen VLM server

Both scripts expose the same environment variables and use the OpenAI-compatible
`/v1` API.  They work for Qwen2.5-VL and Qwen3-VL; set `MODEL_PATH` and
`MODEL_NAME` accordingly.  The default example uses two GPUs.

```bash
MODEL_PATH=/models/Qwen3-VL-8B-Instruct \
MODEL_NAME=qwen3-vl-8b CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 \
bash sglang.sh
```

```bash
MODEL_PATH=/models/Qwen2.5-VL-7B-Instruct \
MODEL_NAME=qwen2.5-vl-7b CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 \
bash vllm.sh
```

Use a long random `API_KEY` for a non-local deployment and pass the same key to
`eval.sh`.  `sglang.sh` uses `python -m sglang.launch_server`; if the installed
SGLang release uses a different entry point, run its equivalent server command
with the same `--served-model-name`, `--port` and tensor-parallel settings.

## Run inference and judge accuracy

`eval.sh` runs one version for all available dataset directories, then runs the
corresponding judge.  Set `MODEL_VER=v0`, `v2` or `v4`; use `DATASET=<directory>`
for one benchmark.  The judge is deterministic for exact/option/yes-no and
grounding answers.  Add `--use_vlm_judge` when the answer requires semantic
checking by the served VLM; the judge prompt contains only the question,
reference answer and model answer (the prediction JSONL still retains the
original image path for auditing).

```bash
cd sCoT-main
MODEL_VER=v4 QWEN_VER=3 \
DATA_ROOT=/home/dataset-assist-0/workspace/deepeyes271/benchmarks \
WORK_DIRS=/data/work_dirs/deepeyes_others \
MODEL_NAME=qwen3-vl-8b \
API_URL=http://127.0.0.1:18938/v1 \
API_KEY=replace-with-a-long-random-key \
bash eval/deepeyes_others/eval.sh
```

For a single run (useful when selecting a split):

```bash
python eval/deepeyes_others/mathvista_mini/eval_mathvista_mini_v4.py \
  --dataset_path /data/deepeyes_others/mathvista_mini \
  --save_path /data/work_dirs/deepeyes_others \
  --model_name qwen3-vl-8b --eval_model_name qwen3-vl-8b \
  --api_url http://127.0.0.1:18938/v1 --api_key replace-with-a-long-random-key \
  --qwen_ver 3 --num_workers 8 --max_samples 0 --split testmini
python eval/deepeyes_others/mathvista_mini/judge_result_mathvista_mini.py \
  --save_path /data/work_dirs/deepeyes_others --model_name qwen3-vl-8b \
  --prediction_file /data/work_dirs/deepeyes_others/qwen3-vl-8b/mathvista_mini/predictions_v4.jsonl
```

The v0/v2/v4 evaluators now follow the corresponding HR-Bench conversation
protocol.  v0 repeatedly sends the crop returned by `image_zoom_in_tool` until
an answer is produced.  v2 parses the initial regions, constructs the same
`k_list`, performs the zoom-out crop and local zoom-in loop with IoU stopping,
then sends the final crop(s) and accumulated thinking for the answer.  v4 does
the same loop while preserving the synthetic `image_zoom_out_tool` messages in
the conversation history.  The prompt variant is selected with
`--prompt_ver 1|2|3` for DeepEyes, Qwen2.5-VL or Qwen3-VL system prompts.

The v2/v4 scripts accept the HR-Bench-compatible arguments
`--hrbench_path` (an alias of `--dataset_path`), `--qwen_ver`, `--prompt_ver`,
`--k`, `--iou_thr`, `--max_iter`, `--max_level` and `--expand_mode`, in
addition to the API/model/save arguments.  `eval.sh` and `eval_simple.sh`
expose the same values through `PROMPT_VER`, `ZOOM_K`, `IOU_THR`, `MAX_ITER`,
`MAX_LEVEL` and `EXPAND_MODE`.

## Files and outputs

Predictions are written to
`<WORK_DIRS>/<MODEL_NAME>/<dataset>/predictions_<version>.jsonl`.  The judge
writes a sibling `<...>_judge.jsonl` and a compact
`judge_result_<version>.json` with `count` and `accuracy`.  Failed API calls are
kept with `status: "error"` and an error string, so a partial run can be
diagnosed before it is repeated.  In addition to the running `accuracy`, POPE
reports the four official subset indicators `Adversarial`, `Popular`, `Random`
and `Overall` in the summary JSON.  The summary also retains precision, recall,
F1 and yes-ratio for compatibility with the usual POPE report.  Every evaluator
shows a sample progress bar (or a line-based fallback when `tqdm` is not
installed) labelled with its dataset and protocol version.

