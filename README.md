# CG-Lab

计算机图形学实验合集。基于 Python + [Taichi](https://github.com/taichi-dev/taichi) 实现，涵盖粒子系统模拟、3D 变换管线等图形学核心算法。

## 环境要求

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) 包管理器

```bash
# 安装依赖
uv sync
```
## 实验列表（持续更新中）

### Work0 - 粒子群交互系统

基于 Taichi GPU 内核的实时粒子模拟系统。1000 个粒子在鼠标引力场中运动，包含空气阻力、边界碰撞与能量损失等物理效果。

```bash
uv run -m src.Work0.main
```

**核心内容：** GPU 并行计算 / 引力场模拟 / 碰撞检测 / 实时渲染

### Work1 - 3D 变换渲染管线

手动实现完整的 MVP（Model-View-Projection）矩阵变换管线，将 3D 几何体经过模型变换、视图变换、透视投影后映射到屏幕空间并绘制。

```bash
uv run -m src.Work1.main
```

**核心内容：** 模型矩阵 / 视图矩阵 / 透视投影 / 齐次坐标除法 / 屏幕映射

**操作：** `A` / `D` 旋转，`ESC` 退出

### Work1_pro - 3D 立方体线框渲染

Work1 的拓展，将几何图元升级为三维正方体，实现完整的 MVP 变换管线与透视线框渲染。支持绕 Y 轴和 X 轴的双轴旋转，不同面的边使用不同颜色区分空间深度。

```bash
uv run -m src.Work1_pro.main
```

**核心内容：** 3D 正方体构建 / 双轴旋转矩阵 / 透视投影 / 线框渲染 / 深度颜色区分

**操作：** `A` / `D` 绕 Y 轴旋转，`W` / `S` 绕 X 轴旋转，`空格` 开关自动旋转，`ESC` 退出

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.12 |
| GPU 框架 | Taichi >= 1.7.4 |
| 包管理 | uv |

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
│   └── Work1_pro/      # 3D 立方体线框渲染（选做拓展）
│       ├── config.py   # 正方体顶点、边、颜色与相机配置
│       ├── transform.py # MVP 矩阵（含双轴旋转）
│       └── main.py     # 线框渲染与交互
├── pyproject.toml
└── README.md
```
