"""
模型推理脚本

使用训练好的模型对视频进行目标检测。
支持实时显示、结果保存、检测统计等功能。
"""

import cv2
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.config import Config


class BasketballDetector:
    """篮球检测模型推理器"""

    def __init__(self, model_path: str = None, config: Config = None):
        """
        初始化检测器

        Args:
            model_path: 模型文件路径
            config: 配置对象
        """
        self.config = config or Config()

        if model_path:
            self.model = YOLO(model_path)
        else:
            # 加载最新训练的模型
            models_dir = self.config.get_path("models_dir")
            best_model = max(
                (models_dir / "basketball_detector").glob("*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
                default=None,
            )
            if best_model:
                self.model = YOLO(str(best_model))
                print(f"加载模型：{best_model}")
            else:
                raise FileNotFoundError(
                    "未找到训练好的模型，请指定模型路径或先训练模型"
                )

        # 检测配置
        detect_config = self.config.get("DETECT", {})
        self.conf_threshold = detect_config.get("conf", 0.25)
        self.iou_threshold = detect_config.get("iou", 0.45)
        self.max_det = detect_config.get("max_det", 300)

        # 类别映射
        self.class_names = self.config.class_names

    def detect_image(self, image_path: str, save: bool = False) -> Dict:
        """
        检测单张图像

        Args:
            image_path: 图像路径
            save: 是否保存结果图像

        Returns:
            检测结果
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            save=save,
        )

        result = results[0]
        boxes = result.boxes

        detection_info = {
            "image": image_path,
            "detections": [],
            "summary": {},
        }

        # 解析检测结果
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            det = {
                "class": self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                "class_id": cls_id,
                "confidence": conf,
                "bbox": bbox,  # [x1, y1, x2, y2]
            }
            detection_info["detections"].append(det)

            # 统计
            cls_name = det["class"]
            detection_info["summary"][cls_name] = (
                detection_info["summary"].get(cls_name, 0) + 1
            )

        return detection_info

    def detect_video(
        self,
        video_path: str,
        output_path: str = None,
        show: bool = False,
        save: bool = True,
    ) -> List[Dict]:
        """
        检测视频

        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            show: 是否实时显示
            save: 是否保存结果视频

        Returns:
            每帧的检测结果列表
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{video_path}")

        # 视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 输出视频
        if save:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"results/detected_{Path(video_path).stem}_{timestamp}.mp4"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None

        print(f"\n开始处理视频:")
        print(f"  输入：{video_path}")
        print(f"  分辨率：{width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  总帧数：{total_frames}")
        if save:
            print(f"  输出：{output_path}")
        print()

        all_detections = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 运行检测
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
            )

            # 绘制结果
            result_frame = results[0].plot()

            # 保存
            if out:
                out.write(result_frame)

            # 显示
            if show:
                cv2.imshow("Detection", result_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 记录检测结果
            boxes = results[0].boxes
            frame_dets = {
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "detections": [],
            }

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                frame_dets["detections"].append(
                    {
                        "class": self.class_names[cls_id]
                        if cls_id < len(self.class_names)
                        else f"class_{cls_id}",
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": bbox,
                    }
                )

            all_detections.append(frame_dets)

            # 进度
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"进度：{frame_idx}/{total_frames} ({progress:.1f}%)")

            frame_idx += 1

        cap.release()
        if out:
            out.release()

        if show:
            cv2.destroyAllWindows()

        print(f"\n处理完成!")
        print(f"  处理帧数：{frame_idx}")
        print(f"  总检测：{sum(len(d['detections']) for d in all_detections)}")

        return all_detections

    def detect_batch(self, image_dir: str, save: bool = False) -> List[Dict]:
        """
        批量检测图像

        Args:
            image_dir: 图像目录
            save: 是否保存结果图像

        Returns:
            检测结果列表
        """
        image_path = Path(image_dir)
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        images = []
        for ext in image_extensions:
            images.extend(image_path.glob(f"*{ext}"))
            images.extend(image_path.glob(f"*{ext.upper()}"))

        if not images:
            print(f"未在 {image_dir} 中找到图像文件")
            return []

        print(f"发现 {len(images)} 张图像，开始检测...")

        all_results = []
        for i, img_path in enumerate(images):
            result = self.detect_image(str(img_path), save=save)
            all_results.append(result)

            if (i + 1) % 10 == 0:
                print(f"进度：{i + 1}/{len(images)}")

        print(f"批量检测完成!")
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="篮球检测模型推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检测单张图像
  python detect.py image test.jpg

  # 检测图像并保存结果
  python detect.py image test.jpg --save

  # 检测视频
  python detect.py video training.mp4

  # 检测视频并保存结果
  python detect.py video training.mp4 --save --output results/output.mp4

  # 批量检测图像目录
  python detect.py batch data/frames

  # 使用自定义模型
  python detect.py video training.mp4 --model models/basketball_detector/epoch100/best.pt

  # 调整置信度阈值
  python detect.py video training.mp4 --conf 0.5
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # image 命令
    image_parser = subparsers.add_parser("image", help="检测单张图像")
    image_parser.add_argument("image", help="图像文件路径")
    image_parser.add_argument("--save", action="store_true", help="保存结果图像")
    image_parser.add_argument(
        "--model", type=str, help="模型文件路径（默认使用最新训练的模型）"
    )
    image_parser.add_argument(
        "--conf", type=float, default=None, help="置信度阈值"
    )

    # video 命令
    video_parser = subparsers.add_parser("video", help="检测视频")
    video_parser.add_argument("video", help="视频文件路径")
    video_parser.add_argument("--save", action="store_true", help="保存结果视频")
    video_parser.add_argument("--show", action="store_true", help="实时显示")
    video_parser.add_argument("--output", type=str, help="输出视频路径")
    video_parser.add_argument(
        "--model", type=str, help="模型文件路径（默认使用最新训练的模型）"
    )
    video_parser.add_argument(
        "--conf", type=float, default=None, help="置信度阈值"
    )

    # batch 命令
    batch_parser = subparsers.add_parser("batch", help="批量检测图像")
    batch_parser.add_argument("image_dir", help="图像目录")
    batch_parser.add_argument("--save", action="store_true", help="保存结果图像")
    batch_parser.add_argument(
        "--model", type=str, help="模型文件路径（默认使用最新训练的模型）"
    )
    batch_parser.add_argument(
        "--conf", type=float, default=None, help="置信度阈值"
    )

    args = parser.parse_args()

    # 加载模型
    model_path = getattr(args, "model", None)

    try:
        detector = BasketballDetector(model_path=model_path)
    except FileNotFoundError as e:
        print(f"\n错误：{e}")
        print("\n请先训练模型：python src/train.py train --data data/basketball.yaml")
        sys.exit(1)

    # 设置置信度阈值
    if hasattr(args, "conf") and args.conf is not None:
        detector.conf_threshold = args.conf

    if args.command == "image":
        result = detector.detect_image(args.image, save=args.save)
        print(f"\n检测结果:")
        print(f"  图像：{args.image}")
        for cls, count in result["summary"].items():
            print(f"  {cls}: {count}")
        for det in result["detections"]:
            print(
                f"  - {det['class']}: {det['confidence']:.2f} bbox={det['bbox']}"
            )

    elif args.command == "video":
        detector.detect_video(
            args.video,
            output_path=args.output if args.save else None,
            show=args.show,
            save=args.save,
        )

    elif args.command == "batch":
        detector.detect_batch(args.image_dir, save=args.save)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
