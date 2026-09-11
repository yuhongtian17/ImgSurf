# Copyright 2024 PRIME team and/or its affiliates
# Licensed under the Apache License, Version 2.0

from .registry import get_reward_manager_cls, register  # noqa: I001
from .batch import BatchRewardManager
from .dapo import DAPORewardManager
from .naive import NaiveRewardManager
from .prime import PrimeRewardManager
from .imgsurf import ImgSurfRewardManager

__all__ = [
    "BatchRewardManager",
    "DAPORewardManager",
    "NaiveRewardManager",
    "PrimeRewardManager",
    "ImgSurfRewardManager",
    "register",
    "get_reward_manager_cls",
]
