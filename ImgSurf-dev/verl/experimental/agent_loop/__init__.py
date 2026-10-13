# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from .agent_loop import AgentLoopBase, AgentLoopManager
from .single_turn_agent_loop import SingleTurnAgentLoop
from .tool_agent_loop import ToolAgentLoop
from recipe.imgsurf.imgsurf_agent_loop import ImgSurfToolAgentLoop

# Install the small worker compatibility bridges at process import time, before
# AgentLoopWorker schedules its first concurrent rollout batch.
ImgSurfToolAgentLoop._install_worker_bridges()

_ = [SingleTurnAgentLoop, ToolAgentLoop, ImgSurfToolAgentLoop]

__all__ = ["AgentLoopBase", "AgentLoopManager"]
