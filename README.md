# CG-Lab

BNU计算机图形学课程个人实验合集。


**姓名：** 陈宝如
**学号：** 202411081011
**专业：** 2024计算机师范

## 实验列表

### Work1 - 粒子群交互系统

基于 Taichi GPU 内核的实时粒子模拟系统。1000 个粒子在鼠标引力场中运动，包含空气阻力、边界碰撞与能量损失等物理效果。

**操作：** 将键鼠在窗口内移动，观察粒子群运动。

### Work2 - 3D 变换渲染管线

手动实现完整的 MVP（Model-View-Projection）矩阵变换管线，将 3D 几何体经过模型变换、视图变换、透视投影后映射到屏幕空间并绘制。

**操作：** `A` / `D` 旋转

### Work2_pro - 3D 立方体线框渲染 + SLERP 旋转插值

Work2 的拓展，将几何图元升级为三维正方体，实现完整的 MVP 变换管线与透视线框渲染。支持绕 Y 轴和 X 轴的双轴旋转，不同面的边使用不同颜色区分空间深度。此外，利用四元数球面线性插值（SLERP）实现两个不同姿态之间的平滑旋转过渡动画。    

**操作：** `A` / `D` 绕 Y 轴旋转，`W` / `S` 绕 X 轴旋转，`空格` 开关自动旋转，`I` 切换 SLERP 插值模式

### Work3 - Bézier 曲线光栅化与交互绘制

基于 Taichi GPU 内核与 NumPy 实现 Bézier 曲线的交互式绘制与光栅化。手动实现 De Casteljau 算法采样曲线，通过 batching 策略批量发往 GPU 并行写入显存。支持超采样反走样、控制点拖拽、Bézier Spline（Catmull-Rom 转换）插值，以及 De Casteljau 包络线动态可视化动画。

**操作：** `鼠标左键` 添加/拖拽控制点，`鼠标右键` 切换 Bézier/Spline 模式，`空格` 开关反走样，`C` 清空画布

### Work4 - Phong & Blinn-Phong 光照模型

基于 Taichi 实现了一个支持 GPU 加速的实时光线追踪渲染器。脱离了传统 CPU 循环与外部模型，在 Taichi Kernel 中利用数学隐式方程直接定义三维几何体，并实现了完整的局部光照模型。在完成基础的 Ambient、Diffuse 和 Specular 计算之上，搭建了现代化的 UI 交互面板，并完成了**硬阴影 (Hard Shadow)** 与 **Blinn-Phong 高光模型升级** 。

**操作：** `鼠标左键` 拖动左上角 UI 面板的滑动条：`Ka` 调节环境光、 `Kd` 调节漫反射光、 `Ks` 调节镜面高光、`N` 调节高光指数，`Toggle Model` 切换 Phong/Blinn-Phong 模式

### Work5 - Whitted-Style 光线追踪（迭代式）

将 Work3 的局部光照升级为完整的 **Whitted-Style 光线追踪**。在单个 Taichi Kernel 内用 `for` 循环 + `throughput`/`final_color` 方案替代传统递归，完美适配 GPU SIMT。场景包含黑白棋盘格无限大地面（漫反射）、红色漫反射球、银色镜面球，并完成了**硬阴影**与 **Shadow Acne 浮点偏移修正**。选做部分进一步加入了**玻璃球（Snell 折射 + TIR 全反射）**与 **MSAA 抗锯齿**。

**操作：** `鼠标左键` 拖动右上角 UI 面板的滑动条：`Light X / Y / Z` 实时移动光源、`Max Bounces` 调节最大弹射次数（1~8）、`AA Samples` 调节 MSAA 每像素采样数（1~8）

### Work6 - 可微渲染与三维网格形变

基于 PyTorch3D 的可微渲染管线，利用多视角剪影作为监督信号，从零开始将单位球面（ico_sphere）形变为目标三维模型（奶牛）。使用 SoftSilhouetteShader 将光栅化边界软化，使剪影误差通过渲染管线反向传播回顶点坐标。通过 Laplacian Smoothing、Edge Loss 和 Normal Consistency 三大正则项保证网格表面光滑。选做部分扩展为形状 + 顶点颜色联合优化，使用 SoftPhong Shader 监督 RGB 渲染，输出带颜色的 .ply 文件。

**操作：** 按顺序执行 `Work5.ipynb` 中的三个 Cell（环境安装 → 必做实验 → 选做实验），中间结果自动保存至 `output_meshes/` 和 `output_textured_meshes/`。

### Work7 - 质点-弹簧模型布料模拟

基于 **质点-弹簧模型** 实现可交互的 3D 布料模拟。布料离散为 N×N 网格质点，相邻质点通过遵循胡克定律的弹簧相连，弹簧拓扑覆盖 **结构 (Structural) / 剪切 (Shear) / 弯曲 (Bending)** 三种类型。独立实现并对比三种经典数值积分方法 —— **显式欧拉**、**半隐式 / 辛欧拉** 与 **隐式欧拉（定点迭代）**，观察它们在同一物理系统下的稳定性差异。每个积分器把“受力计算”与“位置/速度更新”合并在同一个 `@ti.kernel` 内，最小化每帧 Kernel 启动次数。选做部分加入 **弹簧拓扑运行时开关**（独立切换 Shear / Bending 观察形态差异）与 **空间碰撞**，可实时拖动球体观察布料反应。

**操作：** `鼠标右键` 拖拽旋转视角；控制面板：`[*] Explicit / Semi-Implicit / Implicit Euler` 切换积分方法，拖动 `Damping (k_d)` / `Stiffness (k_s)` 调节物理参数；选做版本（`optional.py`）额外含 `[*] Shear / Bending Springs` 拓扑开关、`[*] Pin 4 Corners` 钉点开关、`Sphere Y / Radius` 球体调节滑动条

### Work8 - LBS 蒙皮

基于 **SMPL** 参数化人体模型，手动拆解并复现完整的 **线性混合蒙皮（Linear Blend Skinning, LBS）** 流程：从模板网格 `v_template` 出发，依次完成 **形状校正**（`shapedirs` 线性组合 + `J_regressor` 关节回归）、**姿态校正**（轴角 → 旋转矩阵 → `posedirs` 修正 blend shape）与 **骨骼层级刚体变换 + 顶点权重混合**，并与 `smplx` 官方 `forward()` 做逐顶点对比，验证手写实现的平均/最大绝对误差均为 0。可视化覆盖单关节权重热力图、全关节主导权重分布，以及模板 / 形状 / 姿态 / 蒙皮四阶段对比图。选做部分实现一个**单关节姿态动画**：固定体型让左肘从 0° 平滑旋转到 110°，逐帧渲染并合成循环 GIF，直观观察蒙皮权重区域如何随骨骼运动被平滑带动。

**操作：** 在 `Work7/` 目录下执行 `.venv\Scripts\python.exe main.py`，四阶段图、对比图、权重图、姿态动画 GIF 与摘要自动保存至 `outputs/`；可用 `--joint-id` / `--anim-joint-id` / `--anim-axis` / `--anim-angle` 等参数自定义（详见 `Work7/README.md`）

## 项目结构

```
CG-lab/
├── src/
│   ├── Work1/          # 粒子群交互系统
│   │   ├── config.py   # 物理与渲染参数配置
│   │   ├── physics.py  # Taichi 物理计算内核
│   │   └── main.py     # GUI 与事件循环
│   ├── Work2/          # 3D 变换渲染管线
│   │   ├── config.py   # 相机与投影参数配置
│   │   ├── transform.py # MVP 矩阵实现
│   │   └── main.py     # 变换管线与渲染
│   ├── Work2_pro/      # 3D 立方体线框渲染（选做拓展）
│   │   ├── config.py   # 正方体顶点、边、颜色与相机配置
│   │   ├── transform.py # MVP 矩阵（含双轴旋转）
│   │   └── main.py     # 线框渲染与交互
│   ├── Work3/          # Bézier 曲线光栅化与交互绘制
│   │   ├── config.py   # 窗口、采样数、对象池与颜色配置
│   │   ├── bezier.py   # De Casteljau 算法、包络线、Spline 转换
│   │   └── main.py     # GPU 光栅化内核与交互渲染
│   ├── Work4/          # Phong & Blinn-Phong 光照模型      
│   │   ├── config.py   # 场景配置（分辨率、相机、光源、物体几何参数与颜色、默认材质）
│   │   ├── raytracer.py # 核心算法库（射线与球/圆锥/地面求交、法线梯度计算）
│   │   └── main.py     # 渲染循环、阴影逻辑、UI 交互构建
│   └── Work5/          # Whitted-Style 光线追踪（折射 + MSAA ）    
│       ├── config.py   # 场景配置（分辨率、相机、光源、几何/材质、玻璃 IOR、UI 范围）
│       ├── raytracer.py # 核心算法库（射线-球/平面求交、反射、Snell 折射、棋盘格采样）
│       └── main.py     # 迭代弹射 Kernel、Phong+阴影着色、MSAA、UI 面板
│   └── Work6/          # 可微渲染与三维网格形变与纹理联合优化
│       ├── data/       # 项目数据
│       ├── output_meshes/ # 剪影驱动 mesh 形变的 .obj 中间结果
|       ├── output_textured_meshes/ # 形状 + 纹理联合优化的 .ply 彩色中间结果
|       ├── Work6.ipynb # 主实验 Notebook
|       ├── visualization/ # 视觉三维演示结果
|       └── README.md   # 项目说明文档
│   └── Work7/          # 质点-弹簧模型布料模拟与弹簧拓扑开关 + 球体碰撞    
│       ├── __init__.py # 子模块加载前 ti.init
│       ├── config.py   # 常量配置
│       ├── physics.py  # 仿真字段定义 + 力学 @ti.func
│       ├── solver.py   # 三个 @ti.kernel 积分求解器 + 调度入口
│       ├── main.py     # 初始化 Kernel + GGUI 控制面板与渲染主循环
│       ├── optional.py # 选做：弹簧类型开关 + 球体碰撞
│       ├── gif/        # 项目效果展示 gif
│       └── README.md   # 项目说明文档
│   └── Work8/          # SMPL 参数化人体模型与线性混合蒙皮（LBS）
│       ├── __init__.py 
│       ├── main.py     # SMPL 加载、手写 LBS、四阶段可视化、误差验证与选做姿态动画
│       ├── SMPL_NEUTRAL.pkl # 本地 SMPL neutral 模型（较大，已 gitignore，需自行放置）
│       ├── outputs/    # 阶段图、对比图、权重图、pose_animation.gif 与 summary.txt
│       └── README.md   # 项目说明文档
├── pyproject.toml
└── README.md

```