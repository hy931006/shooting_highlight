# 篮球训练视频分析 - 目标检测模型训练

用于篮球训练视频剪辑的目标检测模型训练工具。
支持检测**篮球**、**篮筐**和**球员**，可用于后续的进球/丢球事件识别。

## 功能特性

- 🎯 多类别检测：篮球、篮筐、球员
- 📹 视频帧提取：支持按间隔、FPS、关键帧提取
- 🤖 AI 预标注：YOLO-World 开放词汇检测，大幅减少人工标注
- 🏷️ 数据标注：集成 LabelImg，支持 YOLO 格式
- 🚀 YOLOv8: 最新的 YOLO 目标检测模型
- 💻 兼容 CPU 和 NVIDIA GPU 推理

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 提取视频帧

```bash
# 从视频中提取帧
python src/extract_frames.py video.mp4 --fps 1 -o data/frames

# 或使用关键帧提取（推荐，减少冗余）
python src/extract_frames.py video.mp4 --keyframes -o data/frames

# 或使用主入口
python main.py extract video.mp4 --keyframes
```

### 3. AI 预标注（可选，强烈推荐）

使用 YOLO-World 进行预标注，可以大幅减少人工标注时间。
YOLO-World 是一个开放词汇检测模型，可以检测任意类别而无需训练。

```bash
# 对提取的帧进行预标注
python main.py pre-label data/frames

# 或者直接从视频提取并预标注（一键完成）
python main.py auto-label-video training.mp4

# 使用更大模型（更高精度，更慢）
python main.py auto-label-video training.mp4 --model-size x

# 调整置信度阈值（降低阈值可以检测到更多目标，但可能增加误检）
python main.py auto-label-video training.mp4 --conf 0.05
```

**预标注后的人工修正流程：**

1. 预标注会自动生成 `.txt` 标注文件
2. 打开 LabelImg，加载图像和标注
3. 检查并修正预标注结果：
   - 删除误检（框选了错误的目标）
   - 调整不准确的框
   - 补漏检的目标
4. 保存修正后的标注

相比纯手工标注，预标注可以节省约 70-80% 的时间。

### 4. 人工修正标注

```bash
# 安装 LabelImg
pip install labelimg

# 启动标注工具
labelimg

# 创建类别文件
python src/label_tool.py classes
```

在 LabelImg 中:
1. 选择图像目录：`data/frames`
2. 选择保存目录：`data/frames`
3. 选择格式：**YOLO**
4. 加载类别文件：`data/obj.names`
5. 打开已有标注文件（`Ctrl+O` 打开对应的.txt 文件）
6. 修正预标注结果，保存

### 5. 整理数据集

```bash
# 验证标注格式
python src/label_tool.py validate

# 整理数据集（划分训练/验证集）
python src/label_tool.py organize

# 创建数据集配置文件
python src/label_tool.py dataset-yaml
```

### 6. 训练模型

```bash
# 开始训练
python src/train.py train --data data/basketball.yaml

# 使用更大模型（更高精度）
python src/train.py train --data data/basketball.yaml --model yolov8l.pt --epochs 200

# 使用 GPU 训练（如果有）
python src/train.py train --data data/basketball.yaml --device 0

# 或使用主入口
python main.py train --epochs 100
```

### 7. 验证模型

```bash
# 验证模型精度
python src/validate.py validate --detailed

# 比较不同模型
python src/validate.py compare --models model1.pt model2.pt
```

### 8. 使用模型检测

```bash
# 检测单张图像
python src/detect.py image test.jpg --save

# 检测视频
python src/detect.py video training.mp4 --save --output results/output.mp4

# 批量检测
python src/detect.py batch data/frames

# 或使用主入口
python main.py detect video.mp4 --save
```

## 项目结构

```
shooting_highlight/
├── data/
│   ├── raw/              # 原始视频
│   ├── frames/           # 提取的帧（标注前）
│   ├── images/
│   │   ├── train/        # 训练图像
│   │   └── val/          # 验证图像
│   ├── labels/
│   │   ├── train/        # 训练标注
│   │   └── val/          # 验证标注
│   └── basketball.yaml   # 数据集配置
├── models/               # 训练好的模型
├── results/              # 检测结果
├── src/
│   ├── config.py         # 配置管理
│   ├── extract_frames.py # 视频帧提取
│   ├── label_tool.py     # 标注工具
│   ├── auto_label.py     # YOLO-World 预标注
│   ├── train.py          # 模型训练
│   ├── detect.py         # 模型推理
│   └── validate.py       # 模型验证
├── config.yaml           # 主配置文件
└── requirements.txt      # Python 依赖
```

## 配置说明

### 模型选择

| 模型 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| yolov8n.pt | 最快 | 较低 | 快速原型 |
| yolov8s.pt | 快 | 中等 | 实时检测 |
| yolov8m.pt | 中等 | 高 | 推荐默认 |
| yolov8l.pt | 慢 | 很高 | 精度优先 |
| yolov8x.pt | 最慢 | 最高 | 离线处理 |

### 训练参数调整

```bash
# 调整批次大小（根据 GPU 显存）
python src/train.py train --data data/basketball.yaml --batch 32

# 增加训练轮数
python src/train.py train --data data/basketball.yaml --epochs 200

# 调整学习率
python src/train.py train --data data/basketball.yaml --lr0 0.001
```

## 数据标注规范

### YOLO 标注格式

每个标注文件 (.txt) 格式：
```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 0=basketball, 1=hoop, 2=player
- 所有值归一化到 [0, 1]

### 标注建议

1. **篮球**: 标注整个篮球，包括模糊/运动模糊的情况
2. **篮筐**: 标注篮筐前沿和篮网连接处
3. **球员**: 标注完整身体，包括被部分遮挡的情况

## 常见问题

### CUDA out of memory

```bash
# 减小 batch size
python src/train.py train --data data/basketball.yaml --batch 8
```

### 检测精度低

1. 增加训练数据量（标注更多帧）
2. 使用更大的模型（yolov8l.pt 或 yolov8x.pt）
3. 增加训练轮数
4. 确保标注质量

### 检测速度慢

1. 使用更小的模型（yolov8n.pt 或 yolov8s.pt）
2. 减小图像尺寸：`--imgsz 416`
3. 导出为 ONNX 或 TensorRT

## 投篮检测与精彩集锦

### 1. 投篮检测

检测视频中的进球/丢球事件：

```bash
# 基本用法（使用 CPU）
python -m src.shot_detector ./video.mp4 --output results/shots.json

# 指定输出路径
python -m src.shot_detector ./video.mp4 -o results/shots.json

# 指定设备
python -m src.shot_detector ./video.mp4 -o results/shots.json -d cpu   # CPU
python -m src.shot_detector ./video.mp4 -o results/shots.json -d 0    # NVIDIA GPU
python -m src.shot_detector ./video.mp4 -o results/shots.json -d directml  # AMD GPU (需要 DirectML 版 PyTorch)

# 调整置信度阈值
python -m src.shot_detector ./video.mp4 -o results/shots.json --conf 0.3
```

**参数说明：**
- `--device, -d`: 推理设备 (cpu, directml, 0, 1, 2, 3)
- `--conf, -c`: 置信度阈值 (默认 0.25)
- `--output, -o`: 输出 JSON 文件路径
- `--model, -m`: 指定模型文件路径

**输出格式 (JSON)：**
```json
{
  "video": "./video.mp4",
  "events": [
    {"type": "made", "timestamp": 12.5, "confidence": 0.95, "trajectory_points": 45},
    {"type": "missed", "timestamp": 25.3, "confidence": 0.42, "trajectory_points": 30}
  ],
  "summary": {
    "total_made": 10,
    "total_missed": 5,
    "duration": 60.0
  }
}
```

### 2. 生成精彩集锦

根据检测结果，截取进球片段并拼接成集锦视频：

```bash
# 基本用法（进球前2秒 + 进球后1秒）
python create_highlight.py

# 指定视频和结果文件
python create_highlight.py ./video.mp4 ./results/shots.json ./highlight.mp4

# 调整截取时长
# 修改 create_highlight.py 中的默认参数：
before = 3.0  # 进球前3秒
after = 2.0   # 进球后2秒
```

### 3. 查看检测结果

```bash
# 查看进球数量
python -c "
import json
with open('results/shots.json') as f:
    data = json.load(f)
made = len([e for e in data['events'] if e['type'] == 'made'])
missed = len([e for e in data['events'] if e['type'] == 'missed'])
print(f'进球: {made}, 丢球: {missed}')
"

# 查看具体时间点
python -c "
import json
with open('results/shots.json') as f:
    data = json.load(f)
for e in data['events']:
    if e['type'] == 'made':
        print(f'进球: {e[\"timestamp\"]:.2f}s')
"
```

## 设备配置

### 安装 GPU 版本 PyTorch

**NVIDIA GPU (CUDA)：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**AMD GPU (DirectML)：**
```bash
pip install torch-directml
# 注意：DirectML 版本需要特定 PyTorch 版本支持
```

### 训练时指定设备

```bash
# 单 GPU
python -m src.train --device 0

# 多 GPU
python -m src.train --device 0,1,2

# CPU（不推荐，太慢）
python -m src.train --device cpu
```

## 下一步

训练完成后，模型可用于：
- 进球事件检测
- 精彩集锦自动生成
- 球员动作分析
- 训练数据统计
- 自动视频剪辑
