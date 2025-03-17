# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from typing import Dict, Any, Tuple, SupportsFloat

import gymnasium

from gymnasium import spaces

from aworld.config.conf import ToolConfig
from aworld.core.action import GymAction
from aworld.core.common import Tools
from aworld.virtual_environments.env_tool import AsyncEnvTool, ToolFactory


class ActionType(object):
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'


@ToolFactory.register(name=Tools.GYM.value, desc="gym classic control game", asyn=True, supported_action=GymAction)
class OpenAIGym(AsyncEnvTool):
    def __init__(self, env_id: str, wrappers: list = [], **kwargs) -> None:
        """Gym environment constructor.

        Args:
            env_id: gym environment full name
            wrappers: gym environment wrapper list
        """
        self.env_id = env_id
        self._render = kwargs.pop('render', False)
        if self._render and 'render_mode' not in kwargs:
            kwargs['render_mode'] = 'human'
        self.env = self._gym_env_wrappers(env_id, wrappers, **kwargs)
        self.action_space = self.env.action_space
        conf = ToolConfig()
        super(OpenAIGym, self).__init__(conf, **kwargs)

    def name(self):
        return "async_openai_gym"

    async def step(self, action: Any, **kwargs) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self._render:
            await self.render()
        action = OpenAIGym.transform_action(action=action)
        state, reward, terminal, truncate, info = self.env.step(action)
        info['env_id'] = self.env_id
        return OpenAIGym.transform_state(state=state), reward, terminal, truncate, info

    async def render(self):
        return self.env.render()

    async def close(self):
        if self.env:
            self.env.close()
        self.env = None

    async def reset(self, *, seed: int | None = None, options: Dict[str, str] | None = None) -> Tuple[Any, Dict[str, Any]]:
        state = self.env.reset()
        return OpenAIGym.transform_state(state), {"env_id": self.env_id}

    def _action_dim(self):
        if isinstance(self.env.action_space, spaces.Discrete):
            self.action_type = ActionType.DISCRETE
            return self.env.action_space.n
        elif isinstance(self.env.action_space, spaces.Box):
            self.action_type = ActionType.CONTINUOUS
            return self.env.action_space.shape[0]
        else:
            raise Exception('unsupported env.action_space: {}'.format(self.env.action_space))

    def _state_dim(self):
        if len(self.env.observation_space.shape) == 1:
            return self.env.observation_space.shape[0]
        else:
            raise Exception('unsupported observation_space.shape: {}'.format(self.env.observation_space))

    def _gym_env_wrappers(self, env_id, wrappers: list = [], **kwargs):
        env = gymnasium.make(env_id, **kwargs)

        if wrappers:
            for wrapper in wrappers:
                env = wrapper(env)

        return env

    @staticmethod
    def transform_state(state: Any):
        if isinstance(state, tuple):
            states = dict()
            for n, state in enumerate(state):
                state = OpenAIGym.transform_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['gym{}-{}'.format(n, name)] = state
                else:
                    states['gym{}'.format(n)] = state
            return states
        elif isinstance(state, dict):
            states = dict()
            for state_name, state in state.items():
                state = OpenAIGym.transform_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['{}-{}'.format(state_name, name)] = state
                else:
                    states['{}'.format(state_name)] = state
            return states
        else:
            return state

    @staticmethod
    def transform_action(action: Any):
        if not isinstance(action, dict):
            return action
        else:
            actions = dict()
            for name, action in action.items():
                if '-' in name:
                    name, inner_name = name.split('-', 1)
                    if name not in actions:
                        actions[name] = dict()
                    actions[name][inner_name] = action
                else:
                    actions[name] = action
            for name, action in actions.items():
                if isinstance(action, dict):
                    actions[name] = OpenAIGym.transform_action(action=action)
            return actions
