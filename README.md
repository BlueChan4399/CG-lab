# CG-lab

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
│   └── Work1/          # 3D 变换渲染管线
│       ├── config.py   # 相机与投影参数配置
│       ├── transform.py # MVP 矩阵实现
│       └── main.py     # 变换管线与渲染
├── pyproject.toml
└── README.md
```
