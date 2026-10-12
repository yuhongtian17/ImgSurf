# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0
"""ImgSurf overlay for the Ray environment used by PPO workers."""

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR


PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    },
}


def get_ppo_ray_runtime_env():
    """Return the upstream PPO environment plus ImgSurf judge settings."""

    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )
    env_vars = PPO_RAY_RUNTIME_ENV["env_vars"].copy()
    # Ray workers must see the same mode switch as the driver. In particular,
    # an external endpoint must suppress the in-process same-weight judge.
    for key in (
        "IMGSURF_JUDGE_BASE_URL",
        "IMGSURF_JUDGE_MODEL",
        "IMGSURF_JUDGE_API_KEY",
        "IMGSURF_JUDGE_TIMEOUT",
        "IMGSURF_JUDGE_MAX_RETRIES",
        "IMGSURF_JUDGE_MAX_PIXELS",
    ):
        if os.environ.get(key) is not None:
            env_vars[key] = os.environ[key]
    runtime_env = {
        "env_vars": env_vars,
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(PPO_RAY_RUNTIME_ENV["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env
