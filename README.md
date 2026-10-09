# Portrait Image Breakdown

人像拍摄现场桌面工具。回答三个问题：

1. **姿态是怎样的？** — 2D 姿态、朝向、取景与构图分析。
2. **该怎样下指令？** — 将分析转化为可直接说出口的现场指导。
3. **对比参考图差在哪？** — 参考图与当前图的姿态、构图差异及调整建议。

## 工作流

```text
打开照片 → 2D 分析 → 现场指令
                 ↓
          加载参考图 → 图片对比 → 调整
```

三个主页面：**2D 分析**、**现场指令**、**图片对比**。

## 功能

- YOLO 姿态检测（17 关键点、多人）
- 动作分类、身体朝向、构图分析
- 统一指导引擎（优先级排序、HOLD/ADJUST 状态）
- 现场指令模式（主指令 + 辅助指令、语气切换、姿态修正辅助线）
- 图片对比（A/B 并排、构图偏差、姿态差异）
- 画布缩放平移、骨架/BBox/构图叠加层
- 可选 3D 重建（独立窗口，不影响主流程）

## 使用

Python **3.12+**：

```bash
uv sync
uv run python main.py
```

CLI 模式：

```bash
uv run python main.py --cli --image path/to/photo.jpg
```

支持格式：`.jpg` `.jpeg` `.png` `.bmp` `.webp`

## 打包

```bash
uv run pyinstaller --noconfirm --clean main.spec
```

输出：`dist/PortraitImageBreakdown/PortraitImageBreakdown.exe`

需要 `model/` 目录中的姿态模型权重。

## 项目结构

```text
core/                  姿态分析、构图、指导引擎、参考图对比
gui/                   桌面界面（2D 分析、现场指令、图片对比、3D 窗口）
reverse_engineering/   可选 3D 推理（与主流程隔离）
assets/                图标与启动资源
tests/                 测试
main.py                入口
```

## 验证

```bash
uv run pytest -q
```