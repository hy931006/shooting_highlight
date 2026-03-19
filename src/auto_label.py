"""
YOLO-World 预标注工具

使用 YOLO-World 进行开放词汇目标检测，自动生成预标注。
大幅减少人工标注工作量。

YOLO-World 可以检测任意类别，无需重新训练，通过文本提示即可检测新类别。
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class YOWorldPreAnnotator:
    """YOLO-World 预标注器"""

    def __init__(
        self,
        classes: List[str] = None,
        model_size: str = "l",
        device: str = "0",
    ):
        """
        初始化预标注器

        Args:
            classes: 检测类别列表
            model_size: 模型大小 (s, m, l, x)
            device: 设备 (cpu, 0, 1, 2, 3)
        """
        try:
            from ultralytics import YOLOWorld
        except ImportError:
            print("错误：需要安装 ultralytics>=8.2.0")
            print("请运行：pip install -U ultralytics")
            sys.exit(1)

        self.model_size = model_size
        self.device = device

        # 加载 YOLO-World 模型
        print(f"加载 YOLO-World-{model_size} 模型...")
        self.model = YOLOWorld(f"yolov8{model_size}-world.pt")

        # 设置检测类别
        if classes is None:
            # 篮球场景默认类别
            self.classes = ["basketball", "basketball hoop", "player", "person"]
        else:
            self.classes = classes

        print(f"检测类别：{self.classes}")
        self.model.set_classes(self.classes)

    def predict_image(
        self,
        image_path: str,
        output_path: str = None,
        conf_threshold: float = 0.1,
        save_label: bool = True,
    ) -> dict:
        """
        预测单张图像并生成 YOLO 格式标注

        Args:
            image_path: 图像路径
            output_path: 标注文件输出路径
            conf_threshold: 置信度阈值
            save_label: 是否保存标注文件

        Returns:
            检测结果
        """
        from ultralytics import YOLO

        # 运行检测
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        # 解析检测结果
        detections = []
        image_width = result.orig_shape[1]
        image_height = result.orig_shape[0]

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 跳过置信度太低的检测
            if conf < conf_threshold:
                continue

            # YOLO 格式：x_center, y_center, width, height (归一化)
            bbox = box.xyxy[0]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox.tolist()

            # 转换为 YOLO 格式
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            # 确保值在 [0, 1] 范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            # 映射到我们的类别 ID
            class_name = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"

            detections.append(
                {
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox_yolo": [x_center, y_center, width, height],
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )

        # 保存标注文件
        if save_label and output_path:
            label_path = Path(output_path).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)

            with open(label_path, "w") as f:
                for det in detections:
                    # YOLO 格式：class_id x_center y_center width height
                    line = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        det["class_id"],
                        *det["bbox_yolo"],
                    )
                    f.write(line)

            print(f"标注已保存：{label_path}")

        return {
            "image": image_path,
            "detections": detections,
            "summary": {
                cls: sum(1 for d in detections if d["class_name"] == cls)
                for cls in set(d["class_name"] for d in detections)
            },
        }

    def predict_batch(
        self,
        image_dir: str,
        output_dir: str = None,
        conf_threshold: float = 0.1,
        extensions: List[str] = None,
    ) -> List[dict]:
        """
        批量预测图像目录

        Args:
            image_dir: 图像目录
            output_dir: 输出目录（默认与图像同目录）
            conf_threshold: 置信度阈值
            extensions: 图像扩展名列表

        Returns:
            检测结果列表
        """
        image_path = Path(image_dir)

        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        images = []
        for ext in extensions:
            images.extend(image_path.glob(f"*{ext}"))
            images.extend(image_path.glob(f"*{ext.upper()}"))

        if not images:
            print(f"错误：在 {image_dir} 中未找到图像文件")
            return []

        print(f"发现 {len(images)} 张图像，开始预标注...\n")

        all_results = []
        output_path = Path(output_dir) if output_dir else image_path

        for i, img_path in enumerate(images):
            label_output = output_path / img_path.name
            result = self.predict_image(
                str(img_path),
                output_path=str(label_output),
                conf_threshold=conf_threshold,
                save_label=True,
            )
            all_results.append(result)

            # 进度显示
            if (i + 1) % 10 == 0 or i == len(images) - 1:
                print(f"进度：{i + 1}/{len(images)}")

        # 统计
        total_dets = sum(len(r["detections"]) for r in all_results)
        print(f"\n预标注完成!")
        print(f"  处理图像：{len(images)}")
        print(f"  总检测数：{total_dets}")

        class_counts = {}
        for r in all_results:
            for cls, count in r["summary"].items():
                class_counts[cls] = class_counts.get(cls, 0) + count

        print(f"  各类别检测:")
        for cls, count in class_counts.items():
            print(f"    {cls}: {count}")

        return all_results

    def predict_video(
        self,
        video_path: str,
        output_dir: str = None,
        conf_threshold: float = 0.1,
        frame_interval: int = 30,
    ) -> List[dict]:
        """
        从视频中提取关键帧并进行预标注

        Args:
            video_path: 视频路径
            output_dir: 输出目录
            conf_threshold: 置信度阈值
            frame_interval: 帧间隔

        Returns:
            检测结果列表
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"错误：无法打开视频 {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = Path(video_path).stem

        # 输出目录
        if output_dir is None:
            output_dir = Path("data/frames") / f"{video_name}_frames"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"视频信息:")
        print(f"  FPS: {fps:.2f}")
        print(f"  总帧数：{total_frames}")
        print(f"  提取间隔：{frame_interval}帧")
        print(f"  输出目录：{output_dir}")
        print()

        all_results = []
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # 保存帧
                timestamp = frame_idx / fps
                image_path = output_dir / f"{video_name}_frame{frame_idx:06d}_t{timestamp:.2f}s.jpg"
                cv2.imwrite(str(image_path), frame)

                # 预标注
                label_output = image_path
                result = self.predict_image(
                    str(image_path),
                    output_path=str(label_output),
                    conf_threshold=conf_threshold,
                    save_label=True,
                )
                all_results.append(result)
                saved_count += 1

                if saved_count % 10 == 0:
                    print(f"已处理：{saved_count}帧")

            frame_idx += 1

        cap.release()

        print(f"\n视频预标注完成!")
        print(f"  提取帧数：{saved_count}")
        print(f"  输出目录：{output_dir}")

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-World 预标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 批量标注图像目录（默认类别：basketball, basketball hoop, player, person）
  python src/auto_label.py batch data/frames

  # 指定自定义类别
  python src/auto_label.py batch data/frames --classes basketball hoop player

  # 调整置信度阈值
  python src/auto_label.py batch data/frames --conf 0.2

  # 从视频提取帧并标注
  python src/auto_label.py video training.mp4 --interval 30

  # 使用更大的模型（更高精度）
  python src/auto_label.py batch data/frames --model-size x

  # 使用 CPU 推理
  python src/auto_label.py batch data/frames --device cpu
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # batch 命令
    batch_parser = subparsers.add_parser("batch", help="批量标注图像目录")
    batch_parser.add_argument("image_dir", help="图像目录")
    batch_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（默认与输入同目录）",
    )
    batch_parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="检测类别列表（默认：basketball basketball hoop player person）",
    )
    batch_parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="置信度阈值（默认：0.1）",
    )
    batch_parser.add_argument(
        "--model-size",
        type=str,
        default="l",
        choices=["s", "m", "l", "x"],
        help="模型大小（默认：l）",
    )
    batch_parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备（默认：0）",
    )

    # video 命令
    video_parser = subparsers.add_parser("video", help="从视频提取帧并标注")
    video_parser.add_argument("video", help="视频文件路径")
    video_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录",
    )
    video_parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="帧提取间隔（默认：30）",
    )
    video_parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="检测类别列表",
    )
    video_parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="置信度阈值（默认：0.1）",
    )
    video_parser.add_argument(
        "--model-size",
        type=str,
        default="l",
        choices=["s", "m", "l", "x"],
        help="模型大小（默认：l）",
    )
    video_parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备（默认：0）",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # 获取类别
    classes = args.classes if args.classes else ["basketball", "basketball hoop", "player", "person"]

    # 创建预标注器
    annotator = YOWorldPreAnnotator(
        classes=classes,
        model_size=args.model_size,
        device=args.device,
    )

    if args.command == "batch":
        annotator.predict_batch(
            image_dir=args.image_dir,
            output_dir=args.output,
            conf_threshold=args.conf,
        )

    elif args.command == "video":
        annotator.predict_video(
            video_path=args.video,
            output_dir=args.output,
            conf_threshold=args.conf,
            frame_interval=args.interval,
        )


if __name__ == "__main__":
    main()
