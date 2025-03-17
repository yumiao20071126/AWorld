# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from typing import Tuple, List

from playwright.sync_api import Page, BrowserContext

from aworld.core.action_factory import ActionFactory
from aworld.core.common import ToolActionModel, ActionResult, Observation
from aworld.logs.util import logger
from aworld.virtual_environments.env_tool import EnvTool, ToolActionExecutor


class BrowserToolActionExecutor(ToolActionExecutor):
    def __init__(self, env_tool: EnvTool[Observation, List[ToolActionModel]] = None):
        super(BrowserToolActionExecutor, self).__init__(env_tool)

    def execute_action(self, actions: List[ToolActionModel], **kwargs) -> Tuple[
        List[ActionResult], Page]:
        """Execute the specified browser action sequence by agent policy.

        Args:
            actions: Tool action sequence.

        Returns:
            Browser page and action result list.
        """
        action_results = []
        page = self.tool.page
        for action in actions:
            action_result, page = self._exec(action, **kwargs)
            action_results.append(action_result)
        return action_results, page

    async def async_execute_action(self, actions: List[ToolActionModel], **kwargs) -> Tuple[
        List[ActionResult], Page]:
        """Execute the specified browser action sequence by agent policy.

        Args:
            actions: Tool action sequence.

        Returns:
            Browser page and action result list.
        """
        action_results = []
        page = self.tool.page
        for action in actions:
            action_result, page = await self._async_exec(action, **kwargs)
            action_results.append(action_result)
        return action_results, page

    def _exec(self, action_model: ToolActionModel, **kwargs):
        action_name = action_model.action_name
        if action_name not in ActionFactory:
            raise ValueError(f'Action {action_name} not found')

        action = ActionFactory(action_name)
        action_result, page = action.act(action_model, page=self.tool.page, browser=self.tool.context, **kwargs)
        logger.info(f"{action_name} execute finished")
        return action_result, page

    async def _async_exec(self, action_model: ToolActionModel, **kwargs):
        action_name = action_model.action_name
        if action_name not in ActionFactory:
            action_name = action_model.tool_name + action_model.action_name
            if action_name not in ActionFactory:
                raise ValueError(f'Action {action_name} not found')

        action = ActionFactory(action_name)
        action_result, page = await action.async_act(action_model, page=self.tool.page, browser=self.tool.context, **kwargs)
        logger.info(f"{action_name} execute finished")
        return action_result, page
