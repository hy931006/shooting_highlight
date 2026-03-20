"""
轨迹分析模块

功能：通过抛物线拟合分析篮球轨迹，判断进球/丢球

设计说明：
- 使用固定机位优化：篮筐位置在视频开头检测一次后固定使用
- 使用 numpy polyfit 拟合抛物线 (y = ax² + bx + c)
- 进球判断：轨迹最高点在篮筐附近，且后续轨迹向下穿过篮筐水平线
- 丢球判断：轨迹与篮筐区域无相交，或相交后继续向上/向外
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class TrajectoryAnalyzer:
    """
    轨迹分析器

    通过抛物线拟合分析篮球轨迹，判断进球/丢球事件
    """

    # 类别名称常量 (与config.yaml一致)
    CLASS_HOOP = 'hoop'

    def __init__(
        self,
        hoop_proximity: float = 50.0,
        min_trajectory_points: int = 5,
        missed_confidence_multiplier: float = 0.6,
        mse_normalization: float = 10000.0,
        confidence_base: float = 0.2,
        confidence_scale: float = 0.8
    ):
        """
        初始化分析器

        Args:
            hoop_proximity: 篮筐附近像素阈值（用于判断球是否经过篮筐附近）
            min_trajectory_points: 最少轨迹点数（少于该点数不进行分析）
            missed_confidence_multiplier: 丢球置信度乘数
            mse_normalization: MSE归一化因子
            confidence_base: 置信度基础值
            confidence_scale: 置信度缩放因子
        """
        self.hoop_proximity = hoop_proximity
        self.min_trajectory_points = min_trajectory_points
        self.missed_confidence_multiplier = missed_confidence_multiplier
        self.mse_normalization = mse_normalization
        self.confidence_base = confidence_base
        self.confidence_scale = confidence_scale

        # 固定机位：篮筐位置
        self.hoop_position: Optional[Dict] = None

    def set_hoop_position(self, bbox: List[float], image_height: int) -> None:
        """
        设置篮筐位置（固定机位优化）

        篮筐位置只需在视频开头检测一次，后续直接使用，
        减少计算量并提高稳定性

        Args:
            bbox: 篮筐边界框 [x1, y1, x2, y2]
            image_height: 图像高度（用于坐标归一化）
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.hoop_position = {
            'x': center_x,
            'y': center_y,
            'bbox': bbox,
            'image_height': image_height
        }

    def detect_hoop(self, detections: List[Dict]) -> Optional[Dict]:
        """
        从检测结果中提取篮筐位置

        Args:
            detections: 检测结果列表

        Returns:
            篮筐位置信息，包含x, y, bbox, confidence
        """
        # 查找篮筐检测结果
        hoop_dets = [d for d in detections if d.get('class') == self.CLASS_HOOP]

        if not hoop_dets:
            return None

        # 取置信度最高的篮筐
        best_hoop = max(hoop_dets, key=lambda x: x.get('confidence', 0))
        bbox = best_hoop['bbox']

        return {
            'x': (bbox[0] + bbox[2]) / 2,
            'y': (bbox[1] + bbox[3]) / 2,
            'bbox': bbox,
            'confidence': best_hoop.get('confidence', 0)
        }

    def analyze_trajectory(self, trajectory: List[Dict]) -> Optional[Dict]:
        """
        分析单条轨迹，返回事件信息

        Args:
            trajectory: 轨迹点列表，每个点包含 x, y, timestamp

        Returns:
            事件信息字典，包含 type(made/missed), timestamp, confidence, trajectory_points
            如果轨迹不符合投篮特征则返回 None
        """
        # 输入验证：确保每个轨迹点都有必需的键
        required_keys = {'x', 'y', 'timestamp'}
        for i, point in enumerate(trajectory):
            if not all(k in point for k in required_keys):
                return None

        # 检查轨迹点数是否足够
        if len(trajectory) < self.min_trajectory_points:
            return None

        # 检查是否已设置篮筐位置
        if self.hoop_position is None:
            return None

        # 提取坐标
        x_coords = [p['x'] for p in trajectory]
        y_coords = [p['y'] for p in trajectory]
        timestamps = [p['timestamp'] for p in trajectory]

        hoop_y = self.hoop_position['y']
        hoop_x = self.hoop_position['x']

        # 抛物线拟合 (y = ax² + bx + c)
        # 使用x坐标作为自变量拟合抛物线
        try:
            coeffs = np.polyfit(x_coords, y_coords, 2)
        except (np.linalg.LinAlgError, ValueError):
            # 拟合失败
            return None

        a, b, c = coeffs

        # 判断是否为投篮轨迹（开口应向下，a < 0）
        if a >= 0:
            # 抛物线开口向上，不是投篮轨迹
            return None

        # 计算抛物线顶点（最高点）
        vertex_x = -b / (2 * a)
        vertex_y = a * vertex_x ** 2 + b * vertex_x + c

        # 获取最高点之前的轨迹点
        before_vertex = [(x, y, t) for x, y, t in zip(x_coords, y_coords, timestamps) if x <= vertex_x]
        # 获取最高点之后的轨迹点
        after_vertex = [(x, y, t) for x, y, t in zip(x_coords, y_coords, timestamps) if x >= vertex_x]

        if not after_vertex:
            # 没有最高点之后的数据，无法判断
            return None

        # 进球判断算法：
        # 1. 轨迹最高点在篮筐附近（y_max < hoop_y + hoop_proximity）
        # 2. 后续存在点 y > hoop_y（球向下穿过篮筐水平线）
        # 3. 球经过篮筐附近（x方向距离在阈值内）

        is_near_hoop = abs(vertex_y - hoop_y) < self.hoop_proximity * 2
        passes_through_hoop = any(y > hoop_y for _, y, _ in after_vertex)

        if is_near_hoop and passes_through_hoop:
            # 进一步判断：球是否在篮筐附近的x坐标处
            near_hoop_x = any(
                abs(x - hoop_x) < self.hoop_proximity
                for x, y, _ in after_vertex if y > hoop_y
            )

            if near_hoop_x:
                # 计算置信度
                confidence = self._calculate_confidence(trajectory, coeffs)

                # 找到穿过篮筐水平线的时间点
                cross_idx = None
                for i, (_, y, _) in enumerate(after_vertex):
                    if y > hoop_y:
                        cross_idx = i
                        break

                if cross_idx is not None:
                    timestamp = after_vertex[cross_idx][2]
                else:
                    # 默认使用最后一点的时间
                    timestamp = timestamps[-1]

                return {
                    'type': 'made',
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'trajectory_points': len(trajectory)
                }

        # 丢球判断：
        # 轨迹与篮筐区域无相交，或相交后继续向上/向外
        # 使用默认的较低置信度
        confidence = self._calculate_confidence(trajectory, coeffs)

        return {
            'type': 'missed',
            'timestamp': timestamps[-1],
            'confidence': confidence * self.missed_confidence_multiplier,
            'trajectory_points': len(trajectory)
        }

    def _calculate_confidence(self, trajectory: List[Dict], coeffs: Tuple[float, float, float]) -> float:
        """
        计算置信度

        基于：轨迹点数、拟合质量

        计算公式：
        - 轨迹点数得分：min(len(trajectory) / 20.0, 1.0) * 0.6
        - 拟合质量得分：(1 - mse / mse_normalization) * 0.4
        - 最终：得分 * confidence_scale + confidence_base

        Args:
            trajectory: 轨迹点列表
            coeffs: 抛物线系数 (a, b, c)

        Returns:
            置信度值，范围 [0, 1]
        """
        # 轨迹点数得分
        points_score = min(len(trajectory) / 20.0, 1.0) * 0.6

        # 拟合质量得分
        x_coords = [p['x'] for p in trajectory]
        y_coords = [p['y'] for p in trajectory]

        y_pred = np.polyval(coeffs, x_coords)
        mse = np.mean((np.array(y_coords) - y_pred) ** 2)
        fit_score = max(0, (1 - mse / self.mse_normalization)) * 0.4

        # 最终置信度
        confidence = (points_score + fit_score) * self.confidence_scale + self.confidence_base

        return confidence

    def analyze_all_trajectories(self, trajectories: List[List[Dict]]) -> List[Dict]:
        """
        分析所有轨迹

        Args:
            trajectories: 轨迹列表，每个轨迹是点的列表

        Returns:
            事件列表，按时间排序
        """
        events = []
        for trajectory in trajectories:
            event = self.analyze_trajectory(trajectory)
            if event is not None:
                events.append(event)

        # 按时间排序
        events.sort(key=lambda x: x['timestamp'])
        return events
