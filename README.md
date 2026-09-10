# Portrait Image Breakdown

面向人像摄影的照片分析与相机反推工具。

这是一个原生 **PySide6** 桌面应用，将 2D 姿态/构图分析、保守的相机参数估计、可编辑的 3D 场景重建，以及参考图拍摄指导整合在一起。

## 功能概览

```text
参考图 ─────┐
            ├─→ 构图 / 姿态偏差 ─→ 拍摄目标
当前图 ─────┘
                     │
                     ├─→ 2D 叠加层
                     ├─→ 相机候选解
                     ├─→ 可编辑 3D 场景
                     ├─→ 图像空间锚点 / PnP 交叉验证
                     └─→ 参考相机假设
```

应用会明确区分 **Observed（观测）**、**Estimated（估计）** 和 **Unknown（未知）**。单张照片无法唯一确定焦距、相机距离或房间的绝对尺度，因此程序会保留候选解或假设，不把它们伪装成精确测量值。

## 当前 V3 工作流

1. 打开参考图和当前照片。
2. 比较姿态、主体比例、中心位置和取景。
3. 使用 2D 目标叠加层调整构图与姿态。
4. 进入 **3D 反向工程**，编辑 Camera 与 Scene anchors。
5. 将图像空间点或平面证据绑定到场景锚点。
6. 查看基于锚点的 PnP 交叉验证和 **Reference Camera Hypothesis**。
7. 添加水平 / 垂直 / 自由参考线约束，可选相机 roll 修正。
8. 应用平面位置约束并实时查看 2D 投影。
9. 保存或恢复版本化 `.pibr.json` 重建会话。
10. 使用 Temporal 工作区对采样序列进行保守的姿态平滑。

3D 检查器会把 **Camera** 和 **Scene anchors** 放在前面，把 **Scene people** 放到底部；**2D Projection Preview** 会高亮当前选中的点或平面，让图像证据始终对应到场景模型。

## 图片导入

主画布空闲时可直接：

- 点击画布打开图片选择器；
- 将图片文件拖入画布后直接开始分析。

支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。

工具栏中的 **打开图片** 仍然提供传统文件选择方式；数据集模式则可以选择一个图片文件夹并逐张切换。

## 项目结构

```text
core/                  2D 分析、姿态、构图、摄影提示
reverse_engineering/   相机几何、重建、参考推理
gui/                   桌面界面与 V3 工作区
model/                 本地 YOLO 姿态模型权重（被 gitignore）
tests/                 确定性回归测试
docs/                  架构与领域说明
assets/app.ico         Windows / Qt 应用图标
main.py                桌面 / CLI 入口
pyproject.toml         Python 包元数据与依赖
uv.lock                锁定环境
```

## 架构

仓库将应用组装、GUI 展示、分析逻辑和重建数学分开：

```text
main.py
  └─ gui.application_runtime
      ├─ gui.application_window
      ├─ gui.main_window
      ├─ gui.field_mode
      ├─ gui.reference_mode
      ├─ gui.anchor_calibration_dialog
      └─ gui.v3_completion

GUI / V3
  gui.reverse_3d_v3
      ├─ gui.reverse_3d_workspace   # 3D 检查器 + 锚点/投影工作区
      ├─ gui.reverse_3d             # 底层 3D 场景/投影绘制
      ├─ gui.reverse_3d_reference_line
      │   └─ 参考线 / 相机匹配控制
      └─ gui.reference_line_calibration
          └─ 交互式线段证据 + roll 修正

反向工程
  reverse_engineering.engine
      └─ reverse_engineering.engine_v2   # 当前 2.5 engine 实现
          ├─ 相机 / 几何 / 投影
          ├─ 深度 / 支撑平面证据
          ├─ 图像空间优化
          ├─ 多人物布局
          └─ 拍摄技术评分

参考重建
  reference_reconstruction
      └─ reference_anchor
  reference_camera
      ├─ reference_reconstruction
      ├─ reference_line_calibration
      └─ scene / scene_anchors
```

更详细的依赖关系和清理路线见 [`docs/DEPENDENCY_GRAPH.md`](docs/DEPENDENCY_GRAPH.md)。

## 姿态模型配置

姿态检测统一由 `core/model_config.py` 管理。默认模型为 **YOLO26m Pose**，程序默认查找：

```text
model/yolo26m-pose.pt
```

内置选择为 `n`、`s`、`m`、`l` 和 `x`。只写模型文件名时会在 `model/` 中查找；显式路径则可以指向其他 checkpoint。

开发时可在启动前设置 `PIB_POSE_MODEL`：

```bash
PIB_POSE_MODEL=x uv run python main.py
PIB_POSE_MODEL=yolo26m-pose.pt uv run python main.py
```

首次使用内置 checkpoint 时，如果本地不存在对应权重，程序会从官方来源下载到 `model/`。模型文件本身不进入仓库；只要网络可用，干净 checkout 即可完成自启动配置。

Ultralytics 当前为 YOLO26 Pose 提供五种尺寸，并使用标准 17 点 COCO 姿态格式。模型目录说明见 [`model/README.md`](model/README.md)。

## 开发与发布

### 开发用户

需要 Python **3.12+**。使用项目自己的环境运行：

```bash
uv sync
uv run python main.py
```

运行确定性测试：

```bash
uv run pytest
```

开发环境可以自行使用 PyInstaller 等工具构建本地发行版。仓库不再维护自动化打包脚本；发布构建所需的模型文件也由发布者自行准备。

### 普通用户

普通用户不需要 Python、uv 或开发依赖。发布版本直接运行发行包中的：

```text
PortraitImageBreakdown.exe
```

发行包需要同时包含 `model/` 目录及所需的 `.pt` 姿态模型权重。

## Windows 图标

项目统一使用 `assets/app.ico` 作为应用图标。它由 Python / Qt 运行时使用；构建 EXE 时需要在 PyInstaller 命令中显式指定：

```bash
python -m PyInstaller --noconfirm --clean --onedir --windowed --name PortraitImageBreakdown --icon=assets/app.ico --add-data "assets;assets" --add-data "model;model" --collect-submodules ultralytics main.py
```

如果 Windows 资源管理器仍显示旧的 Python 图标，请先把 EXE 输出到一个新的目录或换一个新的 EXE 文件名再检查。这可以排除 Windows 对 EXE 图标的缓存影响。

## 证据规则

- **Observed / 观测**：直接由像素、EXIF 或用户输入支持。
- **Estimated / 估计**：由姿态、几何关系、先验、深度或求解器推断。
- **Unknown / 未知**：证据不足，不应当作实测值。

重要例子：

- focal length 与 distance 是候选族，而不是唯一解；
- 单目相对深度只作为软排序信号，不代表房间的绝对尺度；
- camera roll 只有在有独立场景线证据时才接受；
- reference-camera 输出是 delta / hypothesis，不会静默替换当前 SceneCamera。

## 状态

**V2.5：** 当前确定性相机分析流水线已经功能完整。

**V3 Phase 2.5：** 当前桌面端参考重建闭环已经功能完整，包括参考目标、多人物相对布局、可编辑场景锚点、图像空间标定、锚点 PnP 交叉验证、参考相机假设、参考线约束、平面约束、重建会话以及 Temporal 姿态平滑。

后续工作主要集中于更丰富的全身目标求解、多观测证据记录、逐帧相机求解，以及可选的实时相机 / tether 集成。
