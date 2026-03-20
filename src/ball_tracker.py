"""
篮球轨迹追踪模块

功能：追踪篮球在连续帧中的运动轨迹

该模块负责：
1. 追踪篮球在连续帧中的位置
2. 使用欧氏距离匹配相邻帧的篮球位置
3. 过滤短暂消失的检测（遮挡处理）
4. 输出轨迹点列表
"""

from typing import List, Dict, Optional, Tuple
import numpy as np


class BallTracker:
    """
    篮球轨迹追踪器

    使用欧氏距离匹配相邻帧的篮球位置，维护多条轨迹，
    并能处理短暂消失（遮挡）的情况。
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        min_trajectory_points: int = 5,
        max_frame_gap: int = 3
    ):
        """
        初始化追踪器

        Args:
            max_distance: 相邻帧篮球中心点最大距离阈值（像素）
            min_trajectory_points: 最少轨迹点数，少于此点数不计入完整轨迹
            max_frame_gap: 最大允许的帧间隔（用于处理短暂遮挡）
        """
        self.max_distance = max_distance
        self.min_trajectory_points = min_trajectory_points
        self.max_frame_gap = max_frame_gap

        # 存储所有已完成的轨迹
        self.trajectories: List[List[Dict]] = []

        # 当前正在追踪的轨迹
        self.current_trajectory: List[Dict] = []

        # 上一帧篮球位置
        self.last_ball_position: Optional[Tuple[float, float]] = None

        # 上一帧的索引，用于检测帧间隔
        self.last_frame_idx: Optional[int] = None

    def update(self, detections: List[Dict], frame_idx: int, timestamp: float) -> None:
        """
        更新轨迹

        Args:
            detections: 当前帧的检测结果列表
            frame_idx: 帧索引
            timestamp: 时间戳(秒)
        """
        # 提取篮球检测
        basketball_dets = [d for d in detections if d.get('class') == 'basketball']

        if not basketball_dets:
            # 没有检测到篮球
            self._handle_no_detection(frame_idx)
            return

        # 取置信度最高的篮球
        best_ball = max(basketball_dets, key=lambda x: x.get('confidence', 0))
        bbox = best_ball['bbox']

        # 计算边界框中心点
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # 构建轨迹点
        point = {
            'x': float(center_x),
            'y': float(center_y),
            'frame_idx': frame_idx,
            'timestamp': float(timestamp),
            'confidence': float(best_ball.get('confidence', 0))
        }

        # 检查是否与上一帧连续
        if self.last_ball_position is not None and self.last_frame_idx is not None:
            # 计算欧氏距离
            distance = np.sqrt(
                (center_x - self.last_ball_position[0]) ** 2 +
                (center_y - self.last_ball_position[1]) ** 2
            )

            # 计算帧间隔
            frame_gap = frame_idx - self.last_frame_idx

            # 如果距离超过阈值或帧间隔过大，可能是新的投篮
            if distance > self.max_distance or frame_gap > self.max_frame_gap:
                # 保存当前轨迹（如果足够长）
                if len(self.current_trajectory) >= self.min_trajectory_points:
                    self.trajectories.append(self.current_trajectory)
                # 开始新轨迹
                self.current_trajectory = []

        # 添加点到当前轨迹
        self.current_trajectory.append(point)

        # 更新上一帧状态
        self.last_ball_position = (center_x, center_y)
        self.last_frame_idx = frame_idx

    def _handle_no_detection(self, frame_idx: int) -> None:
        """
        处理未检测到篮球的情况

        Args:
            frame_idx: 当前帧索引
        """
        # 如果有帧间隔超过阈值，说明轨迹中断
        if self.last_frame_idx is not None:
            frame_gap = frame_idx - self.last_frame_idx

            if frame_gap > self.max_frame_gap:
                # 轨迹中断，保存当前轨迹
                if len(self.current_trajectory) >= self.min_trajectory_points:
                    self.trajectories.append(self.current_trajectory)
                self.current_trajectory = []
                self.last_ball_position = None
                self.last_frame_idx = None
                return

        # 短暂消失（还在max_frame_gap内），继续等待
        # 不保存当前轨迹，因为可能是遮挡
        if len(self.current_trajectory) >= self.min_trajectory_points:
            self.trajectories.append(self.current_trajectory)

        self.current_trajectory = []
        self.last_ball_position = None
        self.last_frame_idx = None

    def get_current_trajectory(self) -> List[Dict]:
        """
        获取当前正在追踪的轨迹

        Returns:
            当前轨迹点列表的副本
        """
        return self.current_trajectory.copy()

    def get_all_trajectories(self) -> List[List[Dict]]:
        """
        获取所有完整轨迹

        Returns:
            所有轨迹的列表（包含当前未保存的轨迹）
        """
        # 复制已保存的轨迹
        all_trajectories = self.trajectories.copy()

        # 如果当前轨迹足够长，添加到结果中
        if len(self.current_trajectory) >= self.min_trajectory_points:
            all_trajectories.append(self.current_trajectory)

        return all_trajectories

    def finish_current_trajectory(self) -> None:
        """
        手动结束当前轨迹并保存

        用于在视频结束时调用，确保最后的轨迹被保存。
        """
        if len(self.current_trajectory) >= self.min_trajectory_points:
            self.trajectories.append(self.current_trajectory)

        self.current_trajectory = []
        self.last_ball_position = None
        self.last_frame_idx = None

    def reset(self) -> None:
        """
        重置追踪器

        清空所有轨迹和状态，准备开始新的追踪会话。
        """
        self.trajectories = []
        self.current_trajectory = []
        self.last_ball_position = None
        self.last_frame_idx = None

    def get_trajectory_count(self) -> int:
        """
        获取已完成的轨迹数量

        Returns:
            轨迹数量
        """
        count = len(self.trajectories)
        if len(self.current_trajectory) >= self.min_trajectory_points:
            count += 1
        return count

    def get_current_trajectory_length(self) -> int:
        """
        获取当前轨迹的点数

        Returns:
            当前轨迹的点数量
        """
        return len(self.current_trajectory)
