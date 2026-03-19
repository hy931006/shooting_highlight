#!/usr/bin/env python3
"""
篮球训练视频分析 - 主入口

提供统一的命令行接口。
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="篮球训练视频分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
工作流程:
  1. 预标注：    main.py pre-label data/frames
  2. 人工修正：  labelimg (手动修正预标注)
  3. 整理数据集：main.py prepare
  4. 训练模型：  main.py train
  5. 验证模型：  main.py validate
  6. 检测视频：  main.py detect video.mp4

示例:
  # 从视频提取帧并预标注
  main.py extract video.mp4 --keyframes
  main.py pre-label data/frames

  # 或者直接从视频提取并预标注
  main.py auto-label-video video.mp4

  # 然后使用 labelimg 修正标注...
  main.py prepare
  main.py train
  main.py detect video.mp4 --save
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # extract 命令
    extract_parser = subparsers.add_parser(
        "extract", help="从视频中提取帧用于标注"
    )
    extract_parser.add_argument("video", help="视频文件路径")
    extract_parser.add_argument(
        "-o",
        "--output",
        default="data/frames",
        help="输出目录（默认：data/frames）",
    )
    extract_group = extract_parser.add_mutually_exclusive_group(required=True)
    extract_group.add_argument(
        "--interval",
        type=int,
        metavar="N",
        help="每 N 帧提取一帧",
    )
    extract_group.add_argument(
        "--fps",
        type=float,
        metavar="FPS",
        help="按目标 FPS 提取（如 1 表示每秒 1 帧）",
    )
    extract_group.add_argument(
        "--keyframes",
        action="store_true",
        help="提取关键帧（基于帧间差异）",
    )

    # pre-label 命令（YOLO-World 预标注）
    prelabel_parser = subparsers.add_parser(
        "pre-label", help="使用 YOLO-World 进行预标注"
    )
    prelabel_parser.add_argument(
        "image_dir", help="图像目录"
    )
    prelabel_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（默认与输入同目录）",
    )
    prelabel_parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["basketball", "basketball hoop", "player", "person"],
        help="检测类别列表",
    )
    prelabel_parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="置信度阈值（默认：0.1）",
    )
    prelabel_parser.add_argument(
        "--model-size",
        type=str,
        default="l",
        choices=["s", "m", "l", "x"],
        help="模型大小（默认：l）",
    )
    prelabel_parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备（默认：0）",
    )

    # auto-label-video 命令（从视频提取并预标注）
    alv_parser = subparsers.add_parser(
        "auto-label-video", help="从视频提取帧并使用 YOLO-World 预标注"
    )
    alv_parser.add_argument("video", help="视频文件路径")
    alv_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录",
    )
    alv_parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="帧提取间隔（默认：30）",
    )
    alv_parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["basketball", "basketball hoop", "player", "person"],
        help="检测类别列表",
    )
    alv_parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="置信度阈值（默认：0.1）",
    )
    alv_parser.add_argument(
        "--model-size",
        type=str,
        default="l",
        choices=["s", "m", "l", "x"],
        help="模型大小（默认：l）",
    )
    alv_parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备（默认：0）",
    )

    # prepare 命令
    prepare_parser = subparsers.add_parser(
        "prepare", help="准备数据集（验证标注 + 划分训练/验证集）"
    )
    prepare_parser.add_argument(
        "--frames-dir",
        default="data/frames",
        help="标注后的帧图像目录",
    )
    prepare_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认：0.8）",
    )

    # train 命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="模型类型（默认：yolov8m.pt）",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数（默认：100）",
    )
    train_parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批次大小（默认：16）",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练设备（默认：0）",
    )

    # validate 命令
    validate_parser = subparsers.add_parser("validate", help="验证模型")
    validate_parser.add_argument(
        "--model",
        type=str,
        help="模型文件路径（默认使用最新训练的模型）",
    )
    validate_parser.add_argument(
        "--detailed",
        action="store_true",
        help="生成详细报告",
    )

    # detect 命令
    detect_parser = subparsers.add_parser("detect", help="检测视频或图像")
    detect_parser.add_argument(
        "source", help="输入文件（视频或图像）"
    )
    detect_parser.add_argument(
        "--save", action="store_true", help="保存结果"
    )
    detect_parser.add_argument(
        "--show", action="store_true", help="实时显示（仅视频）"
    )
    detect_parser.add_argument(
        "--output", type=str, help="输出文件路径"
    )
    detect_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值（默认：0.25）",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # extract 命令
    if args.command == "extract":
        from src.extract_frames import FrameExtractor

        extractor = FrameExtractor(args.video, args.output)

        if args.interval:
            extractor.extract_by_interval(args.interval)
        elif args.fps:
            extractor.extract_by_fps(args.fps)
        elif args.keyframes:
            extractor.extract_keyframes()

    # pre-label 命令（YOLO-World 预标注）
    elif args.command == "pre-label":
        from src.auto_label import YOWorldPreAnnotator

        annotator = YOWorldPreAnnotator(
            classes=args.classes,
            model_size=args.model_size,
            device=args.device,
        )
        annotator.predict_batch(
            image_dir=args.image_dir,
            output_dir=args.output,
            conf_threshold=args.conf,
        )

    # auto-label-video 命令
    elif args.command == "auto-label-video":
        from src.auto_label import YOWorldPreAnnotator

        annotator = YOWorldPreAnnotator(
            classes=args.classes,
            model_size=args.model_size,
            device=args.device,
        )
        annotator.predict_video(
            video_path=args.video,
            output_dir=args.output,
            conf_threshold=args.conf,
            frame_interval=args.interval,
        )

    # prepare 命令
    elif args.command == "prepare":
        from src.label_tool import validate_annotations, organize_dataset, create_dataset_yaml

        print("步骤 1: 验证标注...")
        validate_annotations(args.frames_dir)

        print("\n步骤 2: 整理数据集...")
        organize_dataset(
            frames_dir=args.frames_dir,
            train_ratio=args.train_ratio,
        )

        print("\n步骤 3: 创建数据集配置...")
        create_dataset_yaml()

        print("\n数据集准备完成!")

    # train 命令
    elif args.command == "train":
        from src.train import BasketballDetectorTrainer
        from src.config import Config

        config = Config()
        trainer = BasketballDetectorTrainer(config)
        trainer.create_model(args.model)

        data_path = str(config.get_path("data_dir") / "basketball.yaml")
        trainer.train(
            data=data_path,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
        )

    # validate 命令
    elif args.command == "validate":
        from src.validate import validate_model

        validate_model(model_path=args.model, detailed=args.detailed)

    # detect 命令
    elif args.command == "detect":
        from src.detect import BasketballDetector
        from pathlib import Path

        detector = BasketballDetector()

        source_path = Path(args.source)
        is_video = source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

        if is_video:
            detector.detect_video(
                args.source,
                output_path=args.output if args.save else None,
                show=args.show,
                save=args.save,
            )
        else:
            result = detector.detect_image(args.source, save=args.save)
            print(f"\n检测结果:")
            for cls, count in result["summary"].items():
                print(f"  {cls}: {count}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
