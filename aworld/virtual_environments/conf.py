# coding: utf-8
import os

from config.conf import ToolConfig, ModelConfig


class BrowserToolConfig(ToolConfig):
    keep_browser_open: bool = True
    private: bool = True
    browse_name: str = "chromium"
    use_browser_executor: bool = False


class AndroidToolConfig(ToolConfig):
    avd_name: str | None = None
    adb_path: str | None = os.path.expanduser('~') + "/Library/Android/sdk/platform-tools/adb"
    emulator_path: str | None = os.path.expanduser('~') + "/Library/Android/sdk/emulator/emulator"
    headless: bool | None = None
    max_retry: int | None = 3
    max_episode_steps: int | None = None


class ImageToolConfig(ToolConfig):
    vision_model: ModelConfig
