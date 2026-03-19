"""
YOLOv8 模型训练脚本

支持篮筐、篮球、球员的多类别目标检测。
针对固定机位篮球训练视频优化精度。
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.config import Config


class BasketballDetectorTrainer:
    """篮球检测模型训练器"""

    def __init__(self, config: Config = None):
        """
        初始化训练器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.model = None

    def create_model(self, model_type: str = None, pretrained: bool = True):
        """
        创建或加载模型

        Args:
            model_type: 模型类型 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            pretrained: 是否使用预训练权重

        Returns:
            YOLO 模型
        """
        if model_type is None:
            model_type = self.config.get("TRAIN.model", "yolov8m.pt")

        if pretrained:
            print(f"加载预训练模型：{model_type}")
            self.model = YOLO(model_type)
        else:
            print(f"从头创建模型：{model_type}")
            self.model = YOLO(model_type, pretrained=False)

        return self.model

    def train(
        self,
        data: str,
        epochs: int = None,
        batch: int = None,
        imgsz: int = None,
        device: str = None,
        **kwargs,
    ):
        """
        训练模型

        Args:
            data: 数据集配置文件路径（YOLO 格式）
            epochs: 训练轮数
            batch: 批次大小
            imgsz: 图像大小
            device: 训练设备 (cpu, 0, 1, 2, 3)
            **kwargs: 其他训练参数

        Returns:
            训练结果
        """
        if self.model is None:
            self.create_model()

        # 从配置获取默认值
        train_config = self.config.get("TRAIN", {})

        epochs = epochs or train_config.get("epochs", 100)
        batch = batch or train_config.get("batch", 16)
        imgsz = imgsz or train_config.get("imgsz", 640)
        device = device or train_config.get("device", "0")

        # 构建训练参数
        train_args = {
            "data": data,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            "workers": train_config.get("workers", 4),
            "optimizer": train_config.get("optimizer", "SGD"),
            "lr0": train_config.get("lr0", 0.01),
            "patience": train_config.get("patience", 50),
            "save_period": train_config.get("save_period", 10),
            "project": str(self.config.get_path("models_dir")),
            "name": "basketball_detector",
            "exist_ok": True,
        }

        # 数据增强配置
        aug_config = train_config.get("augmentation", {})
        if aug_config:
            train_args.update(
                {
                    "hsv_h": aug_config.get("hsv_h", 0.015),
                    "hsv_s": aug_config.get("hsv_s", 0.7),
                    "hsv_v": aug_config.get("hsv_v", 0.4),
                    "flipud": aug_config.get("flipud", 0.0),
                    "fliplr": aug_config.get("fliplr", 0.5),
                    "mosaic": aug_config.get("mosaic", 1.0),
                    "mixup": aug_config.get("mixup", 0.0),
                }
            )

        # 覆盖额外参数
        train_args.update(kwargs)

        print("\n" + "=" * 50)
        print("开始训练")
        print("=" * 50)
        print(f"数据集：{data}")
        print(f"模型：{self.config.get('TRAIN.model', 'yolov8m.pt')}")
        print(f"轮数：{epochs}")
        print(f"批次：{batch}")
        print(f"图像大小：{imgsz}")
        print(f"设备：{device}")
        print("=" * 50 + "\n")

        results = self.model.train(**train_args)

        print("\n" + "=" * 50)
        print("训练完成!")
        print("=" * 50)
        print(f"最佳模型保存位置：{results.best}")
        print(f"最后模型保存位置：{results.last}")

        return results

    def validate(self, data: str = None, split: str = "val"):
        """
        验证模型

        Args:
            data: 数据集配置文件路径
            split: 验证集类型 (val, test, train)

        Returns:
            验证结果
        """
        if self.model is None:
            # 尝试加载最新训练的模型
            models_dir = self.config.get_path("models_dir")
            latest_model = max(
                (models_dir / "basketball_detector").glob("*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
                default=None,
            )

            if latest_model:
                print(f"加载模型：{latest_model}")
                self.model = YOLO(str(latest_model))
            else:
                raise FileNotFoundError("未找到训练好的模型，请先训练模型")

        if data is None:
            data = str(self.config.get_path("data_dir") / "basketball.yaml")

        print("\n" + "=" * 50)
        print("验证模型")
        print("=" * 50)

        metrics = self.model.val(data=data, split=split)

        print("\n" + "=" * 50)
        print("验证结果")
        print("=" * 50)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")

        return metrics

    def export(self, format: str = "onnx", imgsz: int = None):
        """
        导出模型

        Args:
            format: 导出格式 (onnx, torchscript, openvino, engine, coreml, saved_model)
            imgsz: 图像大小

        Returns:
            导出文件路径
        """
        if self.model is None:
            raise ValueError("请先加载或训练模型")

        imgsz = imgsz or self.config.get("TRAIN.imgsz", 640)

        print(f"\n导出模型为 {format} 格式...")
        export_path = self.model.export(format=format, imgsz=imgsz)
        print(f"模型已导出：{export_path}")

        return export_path


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 篮球检测模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练模型
  python train.py train --data data/basketball.yaml

  # 使用自定义参数训练
  python train.py train --data data/basketball.yaml --epochs 200 --batch 32

  # 验证模型
  python train.py val

  # 导出模型为 ONNX
  python train.py export --format onnx

  # 导出模型为 TensorRT
  python train.py export --format engine
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # train 命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--data",
        type=str,
        default="data/basketball.yaml",
        help="数据集配置文件路径",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型类型 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数",
    )
    train_parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="批次大小",
    )
    train_parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="图像大小",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备 (cpu, 0, 1, 2, 3)",
    )
    train_parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="不使用预训练权重，从头训练",
    )

    # val 命令
    val_parser = subparsers.add_parser("val", help="验证模型")
    val_parser.add_argument(
        "--data",
        type=str,
        default="data/basketball.yaml",
        help="数据集配置文件路径",
    )
    val_parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="验证集类型",
    )
    val_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型文件路径（默认使用最新训练的模型）",
    )

    # export 命令
    export_parser = subparsers.add_parser("export", help="导出模型")
    export_parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "openvino", "engine", "coreml", "saved_model"],
        help="导出格式",
    )
    export_parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="图像大小",
    )
    export_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型文件路径（默认使用最新训练的模型）",
    )

    args = parser.parse_args()

    config = Config()
    trainer = BasketballDetectorTrainer(config)

    if args.command == "train":
        if args.model:
            trainer.create_model(args.model, pretrained=not args.no_pretrained)
        trainer.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )

    elif args.command == "val":
        if args.model:
            trainer.model = YOLO(args.model)
        trainer.validate(data=args.data, split=args.split)

    elif args.command == "export":
        if args.model:
            trainer.model = YOLO(args.model)
        else:
            # 加载最新模型
            models_dir = config.get_path("models_dir")
            latest_model = max(
                (models_dir / "basketball_detector").glob("*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
                default=None,
            )
            if latest_model:
                trainer.model = YOLO(str(latest_model))
            else:
                raise FileNotFoundError("未找到训练好的模型")
        trainer.export(format=args.format, imgsz=args.imgsz)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
