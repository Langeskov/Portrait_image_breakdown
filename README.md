# Portrait Image Breakdown

一个为人像拍摄现场设计的桌面工具。它只回答三个问题：

1. **这张图片中的人物姿态是怎样的？** — 通过 2D 姿态、朝向、取景与构图分析回答。
2. **我该怎样向模特下指令？** — 将分析转化为短、具体、可直接说出口的现场指令。
3. **对比图片需要怎样调整？** — 以参考图与当前图的姿态、主体位置和构图差异生成调整建议。

## 工作流

```text
打开当前照片 → 2D 分析 → 现场指令
                     ↓
              加载参考照片 → 图片对比 → 再拍 / 再调整
```

主界面只有三个页面：**2D 分析**、**现场指令**、**图片对比**。这条路径优先服务于拍摄时的即时判断，而非摄影参数的理论推导。

## 可选：3D 重建

**3D 重建**位于 2D 分析页右上方，是独立打开的额外工具。它用于探索图片可能对应的相机与场景假设，不会阻塞、改变或扩展主工作流。

3D 结果属于估计而非测量：单张图片无法唯一确定焦距、距离或空间尺度。应用会区分 **Observed（观测）**、**Estimated（估计）** 与 **Unknown（未知）**。

## 使用

需要 Python **3.12+**：

```bash
uv sync
uv run python main.py
```

支持 `.jpg`、`.jpeg`、`.png`、`.bmp` 和 `.webp`。在 2D 页面可点击画布或直接拖入照片。

## 打包 Windows 发行版

发行包需要包含 `model/` 目录和姿态模型权重：

```bash
uv run pyinstaller --noconfirm --clean main.spec
```

输出位于 `dist/PortraitImageBreakdown/`，其中的 `PortraitImageBreakdown.exe` 为可执行程序。

## 项目结构

```text
core/                  2D 姿态、构图与现场指令
gui/                   三个主页面与可选 3D 窗口
reverse_engineering/   与主流程隔离的可选 3D 推理模块
assets/                应用图标与启动资源
main.py                桌面 / CLI 入口
```

## 验证

```bash
uv run --with pytest pytest -q
```
