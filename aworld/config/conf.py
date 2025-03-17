# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import uuid
from typing import Optional, List

from pydantic import BaseModel


class ModelConfig(BaseModel):
    llm_provider: str = None
    llm_model_name: str | None = None
    llm_temperature: float | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    max_input_tokens: int = 128000

class AgentConfig(BaseModel):
    agent_name: str = None
    max_steps: int = 10
    llm_provider: str = None
    llm_model_name: str | None = None
    llm_num_ctx: int | None = None
    llm_temperature: float | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    max_input_tokens: int = 128000
    max_actions_per_step: int = 10
    include_attributes: List[str] = [
        'title',
        'type',
        'name',
        'role',
        'aria-label',
        'placeholder',
        'value',
        'alt',
        'aria-expanded',
        'data-date-format',
    ]
    message_context: Optional[str] = None
    available_file_paths: Optional[List[str]] = None
    ext: dict | None = None


class TaskConfig(BaseModel):
    task_id: str | None = str(uuid.uuid4())
    task_name: str | None = None
    task: str | None = None
    max_steps: int | None = None
    max_actions_per_step: int = 10
    ext: dict | None = None


class ToolConfig(BaseModel):
    use_vision: bool | None = None
    custom_executor: bool = True
    tool_calling_method: str | None = None
    headless: bool | None = False
    disable_security: bool | None = None
    enable_recording: bool | None = None
    window_w: int | None = None
    window_h: int | None = None
    save_recording_path: str | None = None
    save_trace_path: str | None = None
    save_agent_history_path: str | None = None
    task: str | None = None
    config_data: dict | None = None
    max_retry: int | None = None
    ext: dict | None = None
