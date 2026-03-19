"""
数据标注工具

集成 LabelImg 标注工具，支持 YOLO 格式输出。
提供标注启动、格式转换、数据集划分等功能。
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from src.config import Config


def install_labelimg():
    """
    安装 LabelImg 标注工具

    LabelImg 是一个图形化的图像标注工具，支持 YOLO 格式输出。
    """
    print("安装 LabelImg...")
    print("\n使用方法:")
    print("  1. 运行：labelimg")
    print("  2. 选择图像目录和保存目录")
    print("  3. 在左侧工具栏选择 'YOLO' 格式")
    print("  4. 按 'W' 键绘制 bounding box")
    print("  5. 选择类别（basketball/hoop/player）")
    print("  6. 按 Ctrl+S 保存，按 D 切换到下一张图")
    print("\n快捷键:")
    print("  W: 绘制矩形框")
    print("  A: 上一张图")
    print("  D: 下一张图")
    print("  Ctrl+S: 保存标注")
    print("  Ctrl+R: 更改默认保存目录")
    print("  8: 切换类别")


def create_classes_file(output_path: str = "data/obj.names"):
    """
    创建类别定义文件

    Args:
        output_path: 输出路径
    """
    config = Config()
    classes = config.class_names

    with open(output_path, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")

    print(f"类别文件已创建：{output_path}")
    print(f"类别：{classes}")


def organize_dataset(
    frames_dir: str = "data/frames",
    output_dir: str = "data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    整理数据集为 YOLO 格式

    YOLO 格式要求:
    data/
      images/
        train/  # 训练图像
        val/    # 验证图像
      labels/
        train/  # 训练标注 (txt 文件)
        val/    # 验证标注

    Args:
        frames_dir: 标注后的帧图像目录（包含对应的 txt 标注文件）
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    """
    random.seed(seed)

    frames_path = Path(frames_dir)
    output_path = Path(output_dir)

    # 创建目录结构
    dirs = [
        output_path / "images" / "train",
        output_path / "images" / "val",
        output_path / "labels" / "train",
        output_path / "labels" / "val",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 收集所有图像和标注文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []

    for ext in image_extensions:
        images.extend(frames_path.glob(f"*{ext}"))
        images.extend(frames_path.glob(f"*{ext.upper()}"))

    # 过滤有对应标注文件的图像
    valid_images = []
    for img_path in images:
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            valid_images.append(img_path)
        else:
            print(f"警告：{img_path.name} 没有对应的标注文件，跳过")

    if not valid_images:
        print("错误：没有找到带标注的图像文件")
        print(f"请确保在 {frames_dir} 中有图像和对应的.txt 标注文件")
        return

    # 打乱并划分数据集
    random.shuffle(valid_images)
    split_idx = int(len(valid_images) * train_ratio)

    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]

    # 复制文件
    print(f"\n数据集划分:")
    print(f"  总图像数：{len(valid_images)}")
    print(f"  训练集：{len(train_images)}")
    print(f"  验证集：{len(val_images)}")

    for img_path in train_images:
        shutil.copy(img_path, output_path / "images" / "train" / img_path.name)
        label_path = img_path.with_suffix(".txt")
        shutil.copy(label_path, output_path / "labels" / "train" / label_path.name)

    for img_path in val_images:
        shutil.copy(img_path, output_path / "images" / "val" / img_path.name)
        label_path = img_path.with_suffix(".txt")
        shutil.copy(label_path, output_path / "labels" / "val" / label_path.name)

    print(f"\n数据集整理完成:")
    print(f"  训练图像：{output_path / 'images' / 'train'}")
    print(f"  训练标注：{output_path / 'labels' / 'train'}")
    print(f"  验证图像：{output_path / 'images' / 'val'}")
    print(f"  验证标注：{output_path / 'labels' / 'val'}")


def create_dataset_yaml(
    output_path: str = "data/basketball.yaml",
    classes: List[str] = None,
):
    """
    创建 YOLO 训练用的数据集配置文件

    Args:
        output_path: 输出路径
        classes: 类别列表
    """
    config = Config()

    if classes is None:
        classes = config.class_names

    yaml_content = f"""# 篮球训练视频检测数据集

# 路径（相对于训练脚本的位置）
path: {Path(output_path).parent.absolute()}
train: images/train
val: images/val

# 类别数量
nc: {len(classes)}

# 类别名称
names: {classes}
"""

    with open(output_path, "w") as f:
        f.write(yaml_content)

    print(f"数据集配置文件已创建：{output_path}")


def validate_annotations(frames_dir: str = "data/frames"):
    """
    验证标注文件的格式

    YOLO 格式:
    <class_id> <x_center> <y_center> <width> <height>
    所有值归一化到 [0, 1]

    Args:
        frames_dir: 标注文件目录
    """
    config = Config()
    classes = config.class_names
    frames_path = Path(frames_dir)

    errors = []
    warnings = []

    txt_files = list(frames_path.glob("*.txt"))

    for txt_file in txt_files:
        img_file = txt_file.with_suffix(".jpg")
        if not img_file.exists():
            img_file = txt_file.with_suffix(".png")

        with open(txt_file, "r") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(
                    f"{txt_file.name}:{line_num} - 格式错误，应为 5 个值，实际{len(parts)}个"
                )
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                if class_id < 0 or class_id >= len(classes):
                    errors.append(
                        f"{txt_file.name}:{line_num} - 无效的类别 ID: {class_id}"
                    )

                for name, val in [
                    ("x_center", x_center),
                    ("y_center", y_center),
                    ("width", width),
                    ("height", height),
                ]:
                    if val < 0 or val > 1:
                        errors.append(
                            f"{txt_file.name}:{line_num} - {name}={val} 超出 [0,1] 范围"
                        )

            except ValueError as e:
                errors.append(f"{txt_file.name}:{line_num} - 数值解析错误：{e}")

    print(f"\n标注验证结果:")
    print(f"  检查文件数：{len(txt_files)}")

    if errors:
        print(f"\n错误 ({len(errors)}):")
        for err in errors[:10]:  # 只显示前 10 个错误
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... 还有{len(errors) - 10}个错误")
    else:
        print("  所有标注文件格式正确!")

    if warnings:
        print(f"\n警告 ({len(warnings)}):")
        for warn in warnings[:10]:
            print(f"  {warn}")


def main():
    parser = argparse.ArgumentParser(
        description="数据标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 安装 LabelImg
  python label_tool.py install

  # 创建类别定义文件
  python label_tool.py classes

  # 验证标注格式
  python label_tool.py validate

  # 整理数据集（划分训练/验证集）
  python label_tool.py organize --train-ratio 0.8

  # 创建数据集配置文件
  python label_tool.py dataset-yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # install 命令
    subparsers.add_parser("install", help="安装 LabelImg 标注工具")

    # classes 命令
    subparsers.add_parser("classes", help="创建类别定义文件")

    # validate 命令
    validate_parser = subparsers.add_parser("validate", help="验证标注格式")
    validate_parser.add_argument(
        "--dir",
        default="data/frames",
        help="标注文件目录（默认：data/frames）",
    )

    # organize 命令
    organize_parser = subparsers.add_parser("organize", help="整理数据集为 YOLO 格式")
    organize_parser.add_argument(
        "--frames-dir",
        default="data/frames",
        help="标注后的帧图像目录（默认：data/frames）",
    )
    organize_parser.add_argument(
        "--output-dir",
        default="data",
        help="输出目录（默认：data）",
    )
    organize_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认：0.8）",
    )
    organize_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例（默认：0.2）",
    )
    organize_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）",
    )

    # dataset-yaml 命令
    yaml_parser = subparsers.add_parser("dataset-yaml", help="创建数据集配置文件")
    yaml_parser.add_argument(
        "--output",
        default="data/basketball.yaml",
        help="输出路径（默认：data/basketball.yaml）",
    )

    args = parser.parse_args()

    if args.command == "install":
        install_labelimg()
    elif args.command == "classes":
        create_classes_file()
    elif args.command == "validate":
        validate_annotations(args.dir)
    elif args.command == "organize":
        organize_dataset(
            args.frames_dir,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.seed,
        )
    elif args.command == "dataset-yaml":
        create_dataset_yaml(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
