"""
篮球训练视频分析 - 模型训练包
"""

from .config import Config
from .ball_tracker import BallTracker
from .trajectory_analyzer import TrajectoryAnalyzer
from .shot_detector import ShotDetector

__version__ = "0.1.0"
__all__ = ["Config", "BallTracker", "TrajectoryAnalyzer", "ShotDetector"]
