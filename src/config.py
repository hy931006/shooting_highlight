"""
配置管理模块
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """配置管理类，从 config.yaml 加载配置"""

    def __init__(self, config_path: str = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径，默认为项目根目录的 config.yaml
        """
        if config_path is None:
            # 默认使用项目根目录的 config.yaml
            config_path = Path(__file__).parent.parent / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

        # 创建必要的目录
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_directories(self):
        """创建配置中定义的目录"""
        paths = self.get("PATHS", {})
        for key, path in paths.items():
            if isinstance(path, str):
                # 相对于项目根目录
                full_path = Path(__file__).parent.parent / path
                full_path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点分隔的嵌套键，如 "TRAIN.model"
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_path(self, key: str) -> Path:
        """
        获取路径配置项

        Args:
            key: 配置键

        Returns:
            绝对路径
        """
        path = self.get(key)
        if path is None:
            raise KeyError(f"配置项 {key} 不存在")
        return Path(__file__).parent.parent / path

    @property
    def classes(self) -> list:
        """获取检测类别列表"""
        return self.get("CLASSES", [])

    @property
    def class_names(self) -> list:
        """获取类别名称列表"""
        return [c["name"] for c in self.classes]

    def __repr__(self) -> str:
        return f"Config({self.config_path})"
