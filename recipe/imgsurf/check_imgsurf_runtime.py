#!/usr/bin/env python3
"""Validate the exact ImgSurf/SGLang training runtime before Ray starts."""

from __future__ import annotations

import importlib.util
import sys
from importlib import metadata
from pathlib import Path

from packaging.version import Version


# Prefer the code under the repository receiving the overlay, even when this
# checker is launched by absolute path or from a Ray worker with another cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# Versions coupled to SGLang 0.5.6.post2 and the SCAgent overlay. Keep this in
# sync with requirements_imgsurf.txt and install_imgsurf_env.sh.
EXACT_VERSIONS = {
    "flashinfer-cubin": "0.5.3",
    "flashinfer-python": "0.5.3",
    "openai": "2.6.1",
    "qwen-vl-utils": "0.0.14",
    "ray": "2.50.1",
    "sgl-kernel": "0.3.19",
    "sglang": "0.5.6.post2",
    "tensordict": "0.9.1",
    "torch-memory-saver": "0.0.9",
    "transformers": "4.57.1",
}

# CUDA wheels append a local suffix such as +cu128. Compare their public/base
# version here and validate the compiled CUDA runtime separately.
BASE_VERSIONS = {
    "torch": "2.9.1",
    "torchaudio": "2.9.1",
    "torchvision": "0.24.1",
}


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _check_versions() -> list[str]:
    errors: list[str] = []

    for distribution, expected in EXACT_VERSIONS.items():
        actual = _installed_version(distribution)
        if actual != expected:
            errors.append(f"{distribution}: expected {expected}, found {actual or 'not installed'}")

    for distribution, expected in BASE_VERSIONS.items():
        actual = _installed_version(distribution)
        if actual is None or Version(actual).base_version != expected:
            errors.append(f"{distribution}: expected {expected}[+cu128], found {actual or 'not installed'}")

    return errors


def _check_sglang_api() -> list[str]:
    errors: list[str] = []
    try:
        from sglang.srt.entrypoints.engine import Engine  # noqa: F401
        from sglang.srt.entrypoints.openai.protocol import Tool  # noqa: F401
        from sglang.srt.function_call.function_call_parser import FunctionCallParser  # noqa: F401
        from sglang.srt.managers.io_struct import (
            ReleaseMemoryOccupationReqInput,
            ResumeMemoryOccupationReqInput,
            UpdateWeightsFromTensorReqInput,
        )
        from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: F401
        from sglang.srt.server_args import ServerArgs  # noqa: F401
        from sglang.srt.utils import (
            assert_pkg_version,  # noqa: F401
            get_bool_env_var,  # noqa: F401
            get_local_ip_auto,  # noqa: F401
            get_open_port,  # noqa: F401
            is_cuda,  # noqa: F401
            set_prometheus_multiproc_dir,  # noqa: F401
            set_ulimit,  # noqa: F401
        )

        request_types = (
            ReleaseMemoryOccupationReqInput,
            ResumeMemoryOccupationReqInput,
            UpdateWeightsFromTensorReqInput,
        )
        wrong_modules = [cls.__name__ for cls in request_types if cls.__module__ != "sglang.srt.managers.io_struct"]
        if wrong_modules:
            errors.append("request types are not provided by sglang.srt.managers.io_struct: " + ", ".join(wrong_modules))
    except Exception as exc:  # report the complete import boundary as one actionable error
        errors.append(f"SGLang 0.5.6 API import failed: {type(exc).__name__}: {exc}")

    # Importing weight_sync on a CPU-only Ray driver can trigger optional model
    # backend imports in SGLang 0.5.6. Check that the module exists here; the
    # actual import happens later inside a CUDA worker.
    if importlib.util.find_spec("sglang.srt.weight_sync.utils") is None:
        errors.append("missing module: sglang.srt.weight_sync.utils")

    try:
        from verl.workers.rollout.sglang_rollout.sglang_rollout import AsyncEngine  # noqa: F401
    except Exception as exc:
        errors.append(f"SCAgent SGLang rollout import failed: {type(exc).__name__}: {exc}")

    return errors


def main() -> int:
    errors = _check_versions()
    if not errors:
        try:
            import torch

            if torch.version.cuda != "12.8":
                errors.append(f"torch CUDA runtime: expected 12.8, found {torch.version.cuda or 'none'}")
        except Exception as exc:
            errors.append(f"torch import failed: {type(exc).__name__}: {exc}")

    if not errors:
        errors.extend(_check_sglang_api())

    if errors:
        print("ImgSurf runtime check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "Recreate the clean environment with "
            "'CONDA_ENV_NAME=imgsurf-rl bash recipe/imgsurf/install_imgsurf_env.sh'.",
            file=sys.stderr,
        )
        return 2

    print("ImgSurf runtime check OK: SGLang 0.5.6.post2 / PyTorch 2.9.1+cu128 / Ray 2.50.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
