# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from typing import Any, Dict

from aworld.agents.base import Agent
from aworld.config.conf import AgentConfig


class GymDemoAgent(Agent):
    """Example agent"""

    def __init__(self, conf: AgentConfig, **kwargs):
        super(GymDemoAgent, self).__init__(conf, **kwargs)

    def policy_action(self, observation: Any, info: Dict[str, Any] = None, **kwargs) -> Any:
        import numpy as np

        env_id = info.get("env_id")
        if env_id != 'CartPole-v0':
            raise ValueError("Unsupported env")

        res = np.random.randint(2)
        return res
