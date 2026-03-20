"""
投篮检测主入口

功能：整合YOLO检测、轨迹追踪、轨迹分析，输出进球/丢球事件
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import cv2

from ultralytics import YOLO

from src.ball_tracker import BallTracker
from src.trajectory_analyzer import TrajectoryAnalyzer


class ShotDetector:
    """投篮检测器"""

    # 类别映射
    CLASS_NAMES = {0: 'basketball', 1: 'hoop', 2: 'player'}

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.25,
        hoop_detect_frames: int = 10,
        hoop_confirm_threshold: float = 0.6
    ):
        """
        初始化检测器

        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            hoop_detect_frames: 篮筐检测使用的帧数
            hoop_confirm_threshold: 篮筐确认置信度阈值
        """
        # 加载模型
        if model_path:
            self.model = YOLO(model_path)
        else:
            # 尝试加载默认模型
            default_model = "runs/detect/models/basketball_detector/weights/best.pt"
            if Path(default_model).exists():
                self.model = YOLO(default_model)
            else:
                raise FileNotFoundError("未找到模型文件")

        self.conf_threshold = conf_threshold
        self.hoop_detect_frames = hoop_detect_frames
        self.hoop_confirm_threshold = hoop_confirm_threshold

        # 初始化组件
        self.tracker = BallTracker()
        self.analyzer = TrajectoryAnalyzer()

        # 固定机位：篮筐位置
        self.hoop_position_set = False

    def detect_video(self, video_path: str) -> Dict:
        """
        检测视频中的投篮事件

        Args:
            video_path: 视频文件路径

        Returns:
            检测结果字典
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{video_path}")

        # 视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧, {duration:.1f}秒")

        # 固定机位：先检测篮筐位置
        hoop_detected = self._detect_hoop_position(cap, height)
        cap.release()

        if not hoop_detected:
            print("警告: 未能检测到篮筐位置，使用默认分析逻辑")

        # 重新打开视频进行完整分析
        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # YOLO检测
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False
            )

            # 解析检测结果
            detections = self._parse_detections(results[0])

            # 更新轨迹
            self.tracker.update(detections, frame_idx, timestamp)

            # 进度显示
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"进度: {frame_idx}/{total_frames} ({progress:.1f}%)")

            frame_idx += 1

        cap.release()

        # 分析所有轨迹
        trajectories = self.tracker.get_all_trajectories()
        events = self.analyzer.analyze_all_trajectories(trajectories)

        # 统计
        made_count = sum(1 for e in events if e['type'] == 'made')
        missed_count = sum(1 for e in events if e['type'] == 'missed')

        result = {
            'video': str(video_path),
            'events': events,
            'summary': {
                'total_made': made_count,
                'total_missed': missed_count,
                'duration': duration
            }
        }

        return result

    def _detect_hoop_position(self, cap, image_height: int) -> bool:
        """
        检测篮筐位置（固定机位优化）

        Args:
            cap: 视频捕获对象
            image_height: 图像高度

        Returns:
            是否成功检测
        """
        hoop_positions = []
        frame_count = 0

        while frame_count < self.hoop_detect_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False
            )

            detections = self._parse_detections(results[0])
            hoop = self.analyzer.detect_hoop(detections)

            if hoop and hoop['confidence'] >= self.hoop_confirm_threshold:
                hoop_positions.append(hoop)

            frame_count += 1

        if hoop_positions:
            # 取平均位置
            avg_x = sum(h['x'] for h in hoop_positions) / len(hoop_positions)
            avg_y = sum(h['y'] for h in hoop_positions) / len(hoop_positions)

            self.analyzer.set_hoop_position(
                [avg_x - 50, avg_y - 50, avg_x + 50, avg_y + 50],
                image_height
            )
            print(f"篮筐位置已锁定: ({avg_x:.0f}, {avg_y:.0f})")
            return True

        return False

    def _parse_detections(self, result) -> List[Dict]:
        """解析YOLO检测结果"""
        detections = []
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            cls_name = self.CLASS_NAMES.get(cls_id, f'class_{cls_id}')

            detections.append({
                'class': cls_name,
                'class_id': cls_id,
                'confidence': conf,
                'bbox': bbox
            })

        return detections


def main():
    import argparse

    parser = argparse.ArgumentParser(description='篮球投篮检测')
    parser.add_argument('video', help='视频文件路径')
    parser.add_argument('--model', '-m', help='模型文件路径')
    parser.add_argument('--output', '-o', help='输出JSON文件路径')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='置信度阈值')

    args = parser.parse_args()

    # 创建检测器
    detector = ShotDetector(model_path=args.model, conf_threshold=args.conf)

    # 检测
    result = detector.detect_video(args.video)

    # 输出结果
    print(f"\n检测结果:")
    print(f"  进球: {result['summary']['total_made']}")
    print(f"  丢球: {result['summary']['total_missed']}")

    # 保存JSON
    output_path = args.output
    if output_path is None:
        output_path = f"results/shots_{Path(args.video).stem}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
