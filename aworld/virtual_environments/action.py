# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import abc
from typing import Tuple, Any

from aworld.core.common import ToolActionModel, ActionResult


class ExecutableAction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act(self, action: ToolActionModel, **kwargs) -> Tuple[ActionResult, Any]:
        """"""

    @abc.abstractmethod
    async def async_act(self, action: ToolActionModel, **kwargs) -> Tuple[ActionResult, Any]:
        """"""

