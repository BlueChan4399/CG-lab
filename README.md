# CG-Lab

BNU计算机图形学课程个人实验合集。

## 实验列表（持续更新中）

### Work0 - 粒子群交互系统

基于 Taichi GPU 内核的实时粒子模拟系统。1000 个粒子在鼠标引力场中运动，包含空气阻力、边界碰撞与能量损失等物理效果。

**操作：** 将键鼠在窗口内移动，观察粒子群运动。

### Work1 - 3D 变换渲染管线

手动实现完整的 MVP（Model-View-Projection）矩阵变换管线，将 3D 几何体经过模型变换、视图变换、透视投影后映射到屏幕空间并绘制。

**操作：** `A` / `D` 旋转

### Work1_pro - 3D 立方体线框渲染 + SLERP 旋转插值

Work1 的拓展，将几何图元升级为三维正方体，实现完整的 MVP 变换管线与透视线框渲染。支持绕 Y 轴和 X 轴的双轴旋转，不同面的边使用不同颜色区分空间深度。此外，利用四元数球面线性插值（SLERP）实现两个不同姿态之间的平滑旋转过渡动画。

**操作：** `A` / `D` 绕 Y 轴旋转，`W` / `S` 绕 X 轴旋转，`空格` 开关自动旋转，`I` 切换 SLERP 插值模式

### Work2 - Bézier 曲线光栅化与交互绘制

基于 Taichi GPU 内核与 NumPy 实现 Bézier 曲线的交互式绘制与光栅化。手动实现 De Casteljau 算法采样曲线，通过 batching 策略批量发往 GPU 并行写入显存。支持超采样反走样、控制点拖拽、Bézier Spline（Catmull-Rom 转换）插值，以及 De Casteljau 包络线动态可视化动画。

**操作：** `鼠标左键` 添加/拖拽控制点，`鼠标右键` 切换 Bézier/Spline 模式，`空格` 开关反走样，`C` 清空画布

### Work3 - Phong & Blinn-Phong 光照模型

基于 Taichi 实现了一个支持 GPU 加速的实时光线追踪渲染器。脱离了传统 CPU 循环与外部模型，在 Taichi Kernel 中利用数学隐式方程直接定义三维几何体，并实现了完整的局部光照模型。在完成基础的 Ambient、Diffuse 和 Specular 计算之上，搭建了现代化的 UI 交互面板，并完成了**硬阴影 (Hard Shadow)** 与 **Blinn-Phong 高光模型升级** 。

**操作：** `鼠标左键` 拖动左上角 UI 面板的滑动条：`Ka` 调节环境光、 `Kd` 调节漫反射光、 `Ks` 调节镜面高光、`N` 调节高光指数，`Toggle Model` 切换 Phong/Blinn-Phong 模式

### Work4 - Whitted-Style 光线追踪（迭代式）

将 Work3 的局部光照升级为完整的 **Whitted-Style 光线追踪**。在单个 Taichi Kernel 内用 `for` 循环 + `throughput`/`final_color` 方案替代传统递归，完美适配 GPU SIMT。场景包含黑白棋盘格无限大地面（漫反射）、红色漫反射球、银色镜面球，并完成了**硬阴影**与 **Shadow Acne 浮点偏移修正**。选做部分进一步加入了**玻璃球（Snell 折射 + TIR 全反射）**与 **MSAA 抗锯齿**。

**操作：** `鼠标左键` 拖动右上角 UI 面板的滑动条：`Light X / Y / Z` 实时移动光源、`Max Bounces` 调节最大弹射次数（1~8）、`AA Samples` 调节 MSAA 每像素采样数（1~8）

### Work5 - 可微渲染与三维网格形变

基于 PyTorch3D 的可微渲染管线，利用多视角剪影作为监督信号，从零开始将单位球面（ico_sphere）形变为目标三维模型（奶牛）。使用 SoftSilhouetteShader 将光栅化边界软化，使剪影误差通过渲染管线反向传播回顶点坐标。通过 Laplacian Smoothing、Edge Loss 和 Normal Consistency 三大正则项保证网格表面光滑。选做部分扩展为形状 + 顶点颜色联合优化，使用 SoftPhong Shader 监督 RGB 渲染，输出带颜色的 .ply 文件。

**操作：** 按顺序执行 `Work5.ipynb` 中的三个 Cell（环境安装 → 必做实验 → 选做实验），中间结果自动保存至 `output_meshes/` 和 `output_textured_meshes/`。

## 项目结构

```
CG-lab/
├── src/
│   ├── Work0/          # 粒子群交互系统
│   │   ├── config.py   # 物理与渲染参数配置
│   │   ├── physics.py  # Taichi 物理计算内核
│   │   └── main.py     # GUI 与事件循环
│   ├── Work1/          # 3D 变换渲染管线
│   │   ├── config.py   # 相机与投影参数配置
│   │   ├── transform.py # MVP 矩阵实现
│   │   └── main.py     # 变换管线与渲染
│   ├── Work1_pro/      # 3D 立方体线框渲染（选做拓展）
│   │   ├── config.py   # 正方体顶点、边、颜色与相机配置
│   │   ├── transform.py # MVP 矩阵（含双轴旋转）
│   │   └── main.py     # 线框渲染与交互
│   ├── Work2/          # Bézier 曲线光栅化与交互绘制
│   │   ├── config.py   # 窗口、采样数、对象池与颜色配置
│   │   ├── bezier.py   # De Casteljau 算法、包络线、Spline 转换
│   │   └── main.py     # GPU 光栅化内核与交互渲染
│   ├── Work3/          # Phong & Blinn-Phong 光照模型
│   │   ├── config.py   # 场景配置（分辨率、相机、光源、物体几何参数与颜色、默认材质）
│   │   ├── raytracer.py # 核心算法库（射线与球/圆锥/地面求交、法线梯度计算）
│   │   └── main.py     # 渲染循环、阴影逻辑、UI 交互构建
│   └── Work4/          # Whitted-Style 光线追踪（折射 + MSAA ）
│       ├── config.py   # 场景配置（分辨率、相机、光源、几何/材质、玻璃 IOR、UI 范围）
│       ├── raytracer.py # 核心算法库（射线-球/平面求交、反射、Snell 折射、棋盘格采样）
│       └── main.py     # 迭代弹射 Kernel、Phong+阴影着色、MSAA、UI 面板
│   └── Work5/          # 可微渲染与三维网格形变与纹理联合优化
│       ├── data/       # 项目数据
│       ├── output_meshes/ # 剪影驱动 mesh 形变的 .obj 中间结果
|       ├── output_textured_meshes/ # 形状 + 纹理联合优化的 .ply 彩色中间结果
|       ├── Work5.ipynb # 主实验 Notebook
|       ├── visualization/ # 视觉三维演示结果
|       └── README.md   # 项目说明文档
├── pyproject.toml
└── README.md

```