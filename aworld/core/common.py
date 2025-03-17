# coding: utf-8
from enum import Enum

from pydantic import BaseModel
from typing import Dict, Any, Union, List

from core.dom import DOMElementNode


class ActionResult(BaseModel):
    """Result of executing an action by use tool."""
    is_done: bool = False
    success: bool = None
    content: str = None
    error: str = None
    keep: bool = False


class Tools(Enum):
    """Tool list supported in the framework."""
    BROWSER = "browser"
    ANDROID = "android"
    GYM = "openai_gym"
    SEARCH_API = "search_api"
    SHELL = "shell"
    CODE_EXECUTE = "code_execute"
    FILE = "file"
    IMAGE_ANALYSIS = "image_analysis"


class Agents(Enum):
    """Tool list supported in the framework."""
    BROWSER = "browser_agent"
    ANDROID = "android_agent"
    SEARCH_API = "search_api_agent"
    CODE_EXECUTE = "code_execute_agent"
    FILE = "file_agent"
    IMAGE_ANALYSIS = "image_analysis_agent"
    SHELL = "shell_agent"


class DomTree(BaseModel):
    element_tree: DOMElementNode
    element_map: Dict[int, DOMElementNode]


class Observation(BaseModel):
    dom_tree: Union[str, DomTree] = None
    image: str = None  # base64
    content: Any = None
    action_result: List[ActionResult] = None
    info: Dict[str, Any] = None


class ParamInfo(BaseModel):
    name: str | None = None
    type: str = "str"
    required: bool = False
    desc: str = None
    default_value: Any = None

class ToolActionInfo(BaseModel):
    name: str
    input_params: Dict[str, ParamInfo] = {}
    desc: str = None


class ToolActionModel(BaseModel):
    tool_name: str
    action_name: str
    params: Dict[str, Any] = {}


class Reward(BaseModel):
    value: float
    info: Dict[str, Any]
