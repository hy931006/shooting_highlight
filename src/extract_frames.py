"""
视频帧提取脚本

从篮球训练视频中提取帧用于数据标注。
支持按间隔提取、场景检测提取、关键帧提取等模式。
"""

import cv2
import argparse
from pathlib import Path
from typing import Optional, List
from datetime import datetime


class FrameExtractor:
    """视频帧提取器"""

    def __init__(self, video_path: str, output_dir: str):
        """
        初始化帧提取器

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在：{self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频：{self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        print(f"视频信息:")
        print(f"  路径：{self.video_path}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  总帧数：{self.total_frames}")
        print(f"  时长：{self.duration:.2f}秒")

    def extract_by_interval(self, interval: int = 30, prefix: str = None) -> List[str]:
        """
        按固定间隔提取帧

        Args:
            interval: 提取间隔（帧数），默认每 30 帧提取一帧
            prefix: 输出文件前缀，默认为视频文件名

        Returns:
            保存的文件路径列表
        """
        if prefix is None:
            prefix = self.video_path.stem

        saved_paths = []
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                timestamp = frame_idx / self.fps
                filename = f"{prefix}_frame{frame_idx:06d}_t{timestamp:.2f}s.jpg"
                output_path = self.output_dir / filename
                cv2.imwrite(str(output_path), frame)
                saved_paths.append(str(output_path))
                saved_count += 1

            frame_idx += 1

        self.cap.release()
        print(f"\n提取完成:")
        print(f"  间隔：{interval}帧")
        print(f"  保存：{saved_count}帧")
        print(f"  输出目录：{self.output_dir}")

        return saved_paths

    def extract_by_fps(self, target_fps: float = 1.0, prefix: str = None) -> List[str]:
        """
        按目标 FPS 提取帧（时间维度均匀采样）

        Args:
            target_fps: 目标帧率，默认 1fps（每秒 1 帧）
            prefix: 输出文件前缀

        Returns:
            保存的文件路径列表
        """
        interval = max(1, int(self.fps / target_fps))
        return self.extract_by_interval(interval, prefix)

    def extract_keyframes(self, threshold: float = 30.0, prefix: str = None) -> List[str]:
        """
        提取关键帧（基于帧间差异）

        当相邻帧的差异超过阈值时，保存当前帧。
        适用于只保存有显著变化的帧，减少冗余。

        Args:
            threshold: 帧间差异阈值，越大提取的帧越少
            prefix: 输出文件前缀

        Returns:
            保存的文件路径列表
        """
        if prefix is None:
            prefix = self.video_path.stem

        saved_paths = []
        frame_idx = 0
        saved_count = 0
        prev_frame = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 第一帧总是保存
            if prev_frame is None:
                timestamp = frame_idx / self.fps
                filename = f"{prefix}_frame{frame_idx:06d}_t{timestamp:.2f}s.jpg"
                output_path = self.output_dir / filename
                cv2.imwrite(str(output_path), frame)
                saved_paths.append(str(output_path))
                saved_count += 1
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_idx += 1
                continue

            # 计算帧间差异
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_frame, curr_gray)
            mean_diff = diff.mean()

            if mean_diff > threshold:
                timestamp = frame_idx / self.fps
                filename = f"{prefix}_frame{frame_idx:06d}_t{timestamp:.2f}s.jpg"
                output_path = self.output_dir / filename
                cv2.imwrite(str(output_path), frame)
                saved_paths.append(str(output_path))
                saved_count += 1
                prev_frame = curr_gray

            frame_idx += 1

        self.cap.release()
        print(f"\n关键帧提取完成:")
        print(f"  阈值：{threshold}")
        print(f"  原始帧数：{frame_idx}")
        print(f"  保存帧数：{saved_count}")
        print(f"  压缩率：{saved_count/frame_idx*100:.1f}%")

        return saved_paths

    def extract_at_timestamps(
        self, timestamps: List[float], prefix: str = None
    ) -> List[str]:
        """
        在指定时间点提取帧

        Args:
            timestamps: 时间点列表（秒）
            prefix: 输出文件前缀

        Returns:
            保存的文件路径列表
        """
        if prefix is None:
            prefix = self.video_path.stem

        saved_paths = []

        for ts in sorted(timestamps):
            frame_idx = int(ts * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

            if ret:
                filename = f"{prefix}_t{ts:.2f}s.jpg"
                output_path = self.output_dir / filename
                cv2.imwrite(str(output_path), frame)
                saved_paths.append(str(output_path))

        self.cap.release()
        print(f"\n时间点提取完成:")
        print(f"  保存：{len(saved_paths)}帧")

        return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="从篮球训练视频中提取帧用于数据标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 每 30 帧提取一帧
  python extract_frames.py video.mp4 --interval 30

  # 每秒提取 1 帧（1fps）
  python extract_frames.py video.mp4 --fps 1

  # 提取关键帧（自动检测场景变化）
  python extract_frames.py video.mp4 --keyframes

  # 在指定时间点提取
  python extract_frames.py video.mp4 --timestamps 10.5 30.0 45.5

  # 指定输出目录
  python extract_frames.py video.mp4 -o data/frames --interval 60
        """,
    )

    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument(
        "-o",
        "--output",
        default="data/frames",
        help="输出目录（默认：data/frames）",
    )

    # 提取模式（互斥）
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--interval",
        type=int,
        metavar="N",
        help="每 N 帧提取一帧",
    )
    mode_group.add_argument(
        "--fps",
        type=float,
        metavar="FPS",
        help="按目标 FPS 提取（如 1 表示每秒 1 帧）",
    )
    mode_group.add_argument(
        "--keyframes",
        action="store_true",
        help="提取关键帧（基于帧间差异）",
    )
    mode_group.add_argument(
        "--timestamps",
        type=float,
        nargs="+",
        metavar="T",
        help="在指定时间点提取（秒）",
    )

    # 可选参数
    parser.add_argument(
        "--prefix",
        type=str,
        help="输出文件前缀（默认：视频文件名）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="关键帧提取的帧间差异阈值（默认：30.0）",
    )

    args = parser.parse_args()

    extractor = FrameExtractor(args.video, args.output)

    if args.interval:
        extractor.extract_by_interval(args.interval, args.prefix)
    elif args.fps:
        extractor.extract_by_fps(args.fps, args.prefix)
    elif args.keyframes:
        extractor.extract_keyframes(args.threshold, args.prefix)
    elif args.timestamps:
        extractor.extract_at_timestamps(args.timestamps, args.prefix)


if __name__ == "__main__":
    main()
