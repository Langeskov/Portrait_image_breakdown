# Portrait Image Breakdown

人体姿态分析 & 摄影逆向工程系统

基于 YOLOv26 Pose 的照片分析系统，能够从一张照片中反推拍摄参数、构图意图和摄影手法。

## 功能

### 核心分析 (core/)
- **骨架检测** — YOLOv26 Pose 17关键点 + 扩展点
- **身体朝向** — 正面/背面/左侧/右侧 + 俯仰角
- **动作识别** — 站立/行走/跑步/跳跃/蹲坐等15种动作
- **镜头分析** — 特写/半身/全身/远景 + 仰拍/俯拍/平视
- **构图分析** — 三分法/对称/引导线/留白
- **下一动作建议** — 基于当前状态的智能推荐

### 摄影逆向工程 (reverse_engineering/)
- **透视分析** — 消失点检测、线段收敛、透视强度
- **相机位置估计** — 高度、距离、俯仰角、偏转角
- **焦段估计** — wide/normal/short_telephoto/telephoto 分类 + 35mm等效
- **景深分析** — 前景/背景模糊度、光圈范围推测
- **运动模糊分析** — 模糊类型、方向、快门速度推测
- **摄影手法识别** — 17种摄影技法自动分类
- **反向验证引擎** — 通过虚拟投影优化估计参数
- **摄影动作建议** — MOVE_FORWARD/BACKWARD/ZOOM_IN 等相机操作

### 输出格式
所有估计结果包含：
- 估计值 + 可能范围
- 置信度 (high/medium/low)
- 推断依据
- 不确定性说明

## 运行

`bash
# 安装依赖
pip install ultralytics opencv-python PySide6

# 命令行分析
python main.py --image photo.jpg --cli
python main.py --batch dataset/ --cli

# 图形界面
python main.py
`

## 项目结构

`
photo/
├── main.py                     # 入口 (GUI / CLI / Batch)
├── core/
│   ├── pose_detector.py        # YOLOv26 骨架检测
│   ├── orientation.py          # 身体朝向分析
│   ├── action_classifier.py    # 动作类别识别
│   ├── camera_analyzer.py      # 镜头位置估算
│   ├── composition.py          # 构图分析
│   └── suggestion.py           # 下一动作建议
├── reverse_engineering/
│   ├── data_types.py           # 核心数据结构 (EstimatedValue, Result types)
│   ├── perspective.py          # 透视分析
│   ├── camera_pose.py          # 相机位置估计
│   ├── focal_length.py         # 焦段估计
│   ├── depth_of_field.py       # 景深分析
│   ├── motion_blur.py          # 运动模糊分析
│   ├── shooting_technique.py   # 摄影手法识别
│   ├── simulation.py           # 反向验证引擎
│   └── engine.py               # 主引擎
├── gui/
│   ├── main_window.py          # PySide6 主窗口
│   ├── canvas.py               # 图像画布 + 骨架绘制
│   └── panels.py               # 分析结果面板
└── models/                     # 模型文件 (git ignored)
`

## 技术栈

- Python 3.12
- YOLOv26 (Ultralytics) — 骨架检测
- OpenCV 5.x — 图像处理
- PySide6 — GUI
- NumPy — 数值计算
