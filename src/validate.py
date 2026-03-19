"""
模型验证脚本

评估模型精度，生成可视化报告。
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.config import Config


def validate_model(
    model_path: str = None,
    data: str = None,
    split: str = "val",
    detailed: bool = False,
):
    """
    验证模型精度

    Args:
        model_path: 模型路径
        data: 数据集配置文件
        split: 验证集类型
        detailed: 是否输出详细报告
    """
    config = Config()

    if model_path is None:
        models_dir = config.get_path("models_dir")
        best_model = max(
            (models_dir / "basketball_detector").glob("*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            default=None,
        )
        if best_model:
            model_path = str(best_model)
        else:
            print("错误：未找到训练好的模型")
            print("请先训练模型：python src/train.py train")
            return

    if data is None:
        data = str(config.get_path("data_dir") / "basketball.yaml")

    print(f"加载模型：{model_path}")
    model = YOLO(model_path)

    print(f"\n验证数据集：{data}")
    print(f"验证集：{split}\n")

    # 运行验证
    metrics = model.val(data=data, split=split)

    # 输出结果
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"{'类别':<15} {'数量':<8} {'mAP50':<12} {'mAP50-95':<12}")
    print("-" * 60)

    # 各类别指标
    class_names = config.class_names
    for i, box_metrics in enumerate(metrics.box.maps):
        cls_name = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"{cls_name:<15} {metrics.box.ntotal[i]:<8} {box_metrics[0]:<12.4f} {box_metrics[1]:<12.4f}")

    print("-" * 60)
    print(f"{'总体':<15} {metrics.box.ntotal.sum():<8} {metrics.box.map50:<12.4f} {metrics.box.map:<12.4f}")
    print("=" * 60)

    # 详细报告
    if detailed:
        report = {
            "model": model_path,
            "dataset": data,
            "split": split,
            "date": datetime.now().isoformat(),
            "metrics": {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "classes": {},
            },
        }

        for i, cls_name in enumerate(class_names):
            if i < len(metrics.box.maps):
                report["metrics"]["classes"][cls_name] = {
                    "mAP50": float(metrics.box.maps[i][0]),
                    "mAP50-95": float(metrics.box.maps[i][1]),
                    "count": int(metrics.box.ntotal[i]),
                }

        # 保存报告
        report_dir = Path("results")
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n详细报告已保存：{report_path}")

    return metrics


def compare_models(models: list, data: str = None):
    """
    比较多个模型的性能

    Args:
        models: 模型路径列表
        data: 数据集配置文件
    """
    config = Config()
    if data is None:
        data = str(config.get_path("data_dir") / "basketball.yaml")

    print("\n模型对比")
    print("=" * 80)
    print(f"{'模型':<50} {'mAP50':<12} {'mAP50-95':<12}")
    print("=" * 80)

    results = []
    for model_path in models:
        model = YOLO(model_path)
        metrics = model.val(data=data, split="val", verbose=False)
        results.append((model_path, metrics.box.map50, metrics.box.map))
        print(f"{Path(model_path).name:<50} {metrics.box.map50:<12.4f} {metrics.box.map:<12.4f}")

    print("=" * 80)

    # 找出最佳模型
    best = max(results, key=lambda x: x[2])
    print(f"\n最佳模型：{best[0]} (mAP50-95: {best[2]:.4f})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="模型验证和评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 验证最新模型
  python validate.py

  # 验证指定模型
  python validate.py --model models/basketball_detector/epoch100/weights/best.pt

  # 生成详细报告
  python validate.py --detailed

  # 比较多个模型
  python validate.py compare --models model1.pt model2.pt model3.pt
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # validate 命令
    val_parser = subparsers.add_parser("validate", help="验证模型")
    val_parser.add_argument(
        "--model", type=str, help="模型文件路径（默认使用最新训练的模型）"
    )
    val_parser.add_argument(
        "--data", type=str, default=None, help="数据集配置文件路径"
    )
    val_parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="验证集类型",
    )
    val_parser.add_argument(
        "--detailed", action="store_true", help="生成详细报告"
    )

    # compare 命令
    compare_parser = subparsers.add_parser("compare", help="比较多个模型")
    compare_parser.add_argument(
        "--models", type=str, nargs="+", required=True, help="模型文件路径列表"
    )
    compare_parser.add_argument(
        "--data", type=str, default=None, help="数据集配置文件路径"
    )

    args = parser.parse_args()

    if args.command == "validate" or args.command is None:
        validate_model(
            model_path=args.model,
            data=args.data,
            split=args.split,
            detailed=args.detailed,
        )
    elif args.command == "compare":
        compare_models(args.models, args.data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
