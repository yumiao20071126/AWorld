# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import os

from pydantic import BaseModel

from aworld.core.common import Tools
from aworld.core.task import Task
from aworld.agents.browser.agent import BrowserAgent
from aworld.config.conf import AgentConfig, TaskConfig
from aworld.logs.util import logger
from aworld.virtual_environments.conf import BrowserToolConfig
from aworld.virtual_environments.env_tool import ToolFactory

os.environ['OPENAI_API_KEY'] = ""


# Create the global agent state instance
class BrowserTask(Task):
    def __init__(self, query: str, agent_config: AgentConfig, browser_tool_config: BrowserToolConfig,
                 task_config: TaskConfig, **kwargs):
        super().__init__()  # Removed conf parameter as it's now split into two configs
        self.agent_config = agent_config
        browser_tool_config.headless = False
        self.browser_tool_config = browser_tool_config
        if isinstance(task_config, BaseModel):
            self.task_config = task_config.model_dump()
        else:
            self.task_config = task_config
        self.query = query
        for k, v in kwargs.items():
            setattr(self, k, v)

    # async def run(self):
    #     return asyncio.run(self._run(self.query, **self.kwargs))

    def run(self, query=None):
        if not query:
            query = self.query

        # Initialize browserAgent using agent_config
        browser_agent = BrowserAgent(
            input=query,
            conf=self.agent_config
        )

        # Initialize the browser tool using browser_tool_config
        browser_tool = ToolFactory(Tools.BROWSER.value, conf=self.browser_tool_config)

        # Reset the browser tool
        observation, info = browser_tool.reset()

        # Get the max steps from the task config
        max_steps = self.task_config.get("max_steps", 100)
        step_count = 0

        try:
            while step_count < max_steps:
                # Get action from agent's policy
                action = browser_agent.policy_action(
                    observation=observation,
                    info=info
                )

                # Execute action using browser tool and unpack all return values
                observation, reward, terminated, truncated, info = browser_tool.step(action)

                # Check if there's an exception in info
                if info.get("exception"):
                    logger.error(f"Step {step_count} failed with exception: {info['exception']}")
                    break

                # Check if task should end (either terminated or truncated)
                if browser_tool.finished:
                    break

                step_count += 1

            return {
                "observation": observation,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
                "steps_taken": step_count,
                "success": terminated and info.get("exception") is None
                # Success only if terminated normally without exception
            }

        except Exception as e:
            import traceback
            logger.error(f"Task execution failed with error: {str(e)}\n{traceback.format_exc()}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "steps_taken": step_count,
                "success": False
            }

        finally:
            # Cleanup if not keeping browser open
            if not self.browser_tool_config.keep_browser_open:
                browser_tool.close()


if __name__ == '__main__':
    browser_tool_config = BrowserToolConfig(window_w=1280, window_h=720, keep_browser_open=True)

    agent_config = AgentConfig(
        agent_name="yishan_test",
        llm_provider="antgpt",
        llm_model_name="gpt-4o",
        llm_num_ctx=32000,
        llm_temperature=1
    )
    # agent_config = AgentConfig(
    #     agent_name="yishan_test",
    #     llm_provider="openai",
    #     llm_model_name="gpt-4o",
    #     llm_num_ctx=32000,
    #     llm_temperature=1
    # )

    task_config: TaskConfig = {
        'max_steps': 100,
        'max_actions_per_step': 100
    }

    # BrowserTask("go to google.com and type 'AntGroup' click search and give me the first url",
    #             agent_config=agent_config,
    #             browser_tool_config=browser_tool_config,
    #             task_config=task_config
    #             ).run()
    BrowserTask("go to google.com and type '支付宝' click search and click the first search result",
                agent_config=agent_config,
                browser_tool_config=browser_tool_config,
                task_config=task_config
                ).run()
