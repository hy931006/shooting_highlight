# 投篮检测优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 优化投篮检测算法，提高进球检测准确率，将误判率从261/305降到合理水平

**Architecture:** 修改 TrajectoryAnalyzer 模块的参数和判定逻辑，增大篮筐判断阈值，放宽进球判定条件

**Tech Stack:** Python, NumPy, OpenCV

---

### Task 1: 修改 TrajectoryAnalyzer 参数默认值

**Files:**
- Modify: `src/trajectory_analyzer.py:27-52`

- [ ] **Step 1: 修改 hoop_proximity 默认值**

将 `hoop_proximity` 从 50.0 改为 80.0

```python
def __init__(
    self,
    hoop_proximity: float = 80.0,  # 从 50.0 改为 80.0
    min_trajectory_points: int = 5,
    missed_confidence_multiplier: float = 0.6,
    mse_normalization: float = 10000.0,
    confidence_base: float = 0.2,
    confidence_scale: float = 0.8
):
```

- [ ] **Step 2: 提交代码**

```bash
git add src/trajectory_analyzer.py
git commit -m "feat: increase hoop_proximity threshold to 80px"
```

---

### Task 2: 优化进球判定逻辑

**Files:**
- Modify: `src/trajectory_analyzer.py:165-180`

- [ ] **Step 1: 放宽进球判定的x方向阈值**

将原来的 `hoop_proximity` 改为 `hoop_proximity * 1.5`，使判定更宽松

当前代码 (Line 170-178):
```python
is_near_hoop = abs(vertex_y - hoop_y) < self.hoop_proximity * 2
passes_through_hoop = any(y > hoop_y for _, y, _ in after_vertex)

if is_near_hoop and passes_through_hoop:
    # 进一步判断：球是否在篮筐附近的x坐标处
    near_hoop_x = any(
        abs(x - hoop_x) < self.hoop_proximity
        for x, y, _ in after_vertex if y > hoop_y
    )
```

修改后:
```python
is_near_hoop = abs(vertex_y - hoop_y) < self.hoop_proximity * 2
passes_through_hoop = any(y > hoop_y for _, y, _ in after_vertex)

if is_near_hoop and passes_through_hoop:
    # 放宽判定：球在篮筐附近即可，不需要精确命中
    near_hoop_x = any(
        abs(x - hoop_x) < self.hoop_proximity * 1.5  # 从 hoop_proximity 改为 hoop_proximity * 1.5
        for x, y, _ in after_vertex if y > hoop_y
    )
```

- [ ] **Step 2: 提交代码**

```bash
git add src/trajectory_analyzer.py
git commit -m "feat: relax made shot x-axis criteria"
```

---

### Task 3: 测试优化效果

**Files:**
- Test: 使用 SDIT4229.MP4 视频

- [ ] **Step 1: 运行检测**

```bash
.venv/Scripts/python -m src.shot_detector ./SDIT4229.MP4 --output results/shots_SDIT4229_v3.json
```

- [ ] **Step 2: 检查结果**

```bash
.venv/Scripts/python -c "
import json
with open('results/shots_SDIT4229_v3.json') as f:
    data = json.load(f)
made = len([e for e in data['events'] if e['type'] == 'made'])
missed = len([e for e in data['events'] if e['type'] == 'missed'])
print(f'进球: {made}, 丢球: {missed}')
"
```

- [ ] **Step 3: 分析结果**

如果进球数明显增加（如从44增加到100+），说明优化有效。如果仍不够，考虑方案B。

- [ ] **Step 4: 提交测试结果**

```bash
git add results/shots_SDIT4229_v3.json
git commit -m "test: run optimized detection on SDIT4229 video"
```

---

### Task 4 (如果方案A不足): 方案B - 改进轨迹分析算法

**Files:**
- Modify: `src/trajectory_analyzer.py`

如果 Task 3 结果仍不理想（进球数 < 150），执行此任务：

- [ ] **Step 1: 增加备选判定逻辑**

在现有的 `analyze_trajectory` 方法末尾，增加一个备选判定：

```python
# 在 return {'type': 'missed', ...} 之前添加：

# 备选判定：如果有明显的上升-下降模式，可能是进球
if a < 0 and len(trajectory) >= 10:
    # 检查最高点是否接近篮筐高度
    if abs(vertex_y - hoop_y) < self.hoop_proximity * 2.5:
        # 这是可能的进球
        return {
            'type': 'made',
            'timestamp': timestamps[-1],
            'confidence': self._calculate_confidence(trajectory, coeffs) * 0.8,
            'trajectory_points': len(trajectory)
        }
```

- [ ] **Step 2: 重新测试**

运行 Task 3 的测试命令检查结果

- [ ] **Step 3: 提交**

```bash
git add src/trajectory_analyzer.py
git commit -m "feat: add fallback made shot detection logic"
```
