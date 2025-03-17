# coding: utf-8
import asyncio
import time
import numpy as np
from typing import Any, Dict

from agents.base import Agent
from config.conf import AgentConfig
from core.task import Task
from logs.util import logger
from virtual_environments.gym.openai_gym import OpenAIGym
from virtual_environments.gym.async_openai_gym import OpenAIGym as AOpenAIGym


class GymAgent(Agent):
    """Example agent"""

    def __init__(self, conf: AgentConfig, **kwargs):
        super(GymAgent, self).__init__(conf, **kwargs)

    def policy_action(self, observation: Any, info: Dict[str, Any] = None, **kwargs) -> Any:
        env_id = info.get("env_id")
        if env_id != 'CartPole-v0':
            raise ValueError("Unsupported env")

        res = np.random.randint(2)
        return res


class GymTask(Task):
    def __init__(self, conf):
        super(GymTask, self).__init__(conf)

    def run(self):
        run_gym_game(self.conf.get("env_tool_id"),
                     render_mode=self.conf.get("render_mode", "human"))


def run_gym_game(gym_env_tool_id: str, wrappers: list = [], **kwargs):
    gym_tool = OpenAIGym(gym_env_tool_id, wrappers, **kwargs)
    agent = GymAgent(AgentConfig())
    print('observation space: {}'.format(gym_tool.env.observation_space))
    print('action space: {}'.format(gym_tool.env.action_space))
    print('rende mode: {}'.format(gym_tool.env.render_mode))

    try:
        # init env tool state
        state, info = gym_tool.reset()
        while True:
            # render
            gym_tool.render()
            # agent policy action, also can use llm, only an example
            action = agent.policy_action(state, info)
            print(f"action: ", action)
            # env tool state and reward info based on action
            state, reward, done, truncated, info = gym_tool.step(action=action)
            print('state: {0}; reward: {1}'.format(state, reward))

            if done:
                logger.info("game done!")
                break
            time.sleep(1)
    finally:
        gym_tool.close()


async def async_run_gym_game(gym_env_tool_id: str, wrappers: list = [], **kwargs):
    gym_tool = AOpenAIGym(gym_env_tool_id, wrappers, **kwargs)
    agent = GymAgent(AgentConfig())
    print('observation space: {}'.format(gym_tool.env.observation_space))
    print('action space: {}'.format(gym_tool.env.action_space))
    print('rende mode: {}'.format(gym_tool.env.render_mode))

    try:
        # init env tool state
        state, info = await gym_tool.reset()
        while True:
            # render
            await gym_tool.render()
            # agent policy action, also can use llm, only an example
            action = agent.policy_action(state, info)
            print(f"action: ", action)
            # env tool state and reward info based on action
            state, reward, done, truncated, info = await gym_tool.step(action=action)
            print('state: {0}; reward: {1}'.format(state, reward))

            if done:
                logger.info("game done!")
                break
            time.sleep(1)
    finally:
        await gym_tool.close()


if __name__ == "__main__":
    # We can run the task use utility method, as follows:
    run_gym_game('CartPole-v0', render_mode='human')

    # when solving complex scenarios, can also run tasks like this:
    task = GymTask({"env_tool_id": 'CartPole-v0', "render_mode": 'human'})
    task.run()

    # async run gym
    asyncio.run(async_run_gym_game('CartPole-v0', render_mode='human'))
