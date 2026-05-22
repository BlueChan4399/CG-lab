# CG-Lab6 - Mass-Spring Cloth Simulation（含选做）

## 1. 实验简介

本实验基于 **质点-弹簧模型 (Mass-Spring Model)** 实现可交互的 3D 布料模拟。
布料被离散成 N×N 网格状的质点集合，相邻质点之间通过遵循胡克定律的弹簧相连。
我们独立实现并对比三种经典的数值积分方法 —— **显式欧拉**、**半隐式 / 辛欧拉**
与 **隐式欧拉（定点迭代）** —— 观察它们在同一物理系统下的稳定性差异，并通过
Taichi GGUI 构建实时控制面板用于方法切换与参数实时调节。

核心模块：

- **质点-弹簧模型**：完整实现胡克弹力 + 阻尼力，弹簧拓扑覆盖 **结构 / 剪切 / 弯曲** 三种类型。
- **三种数值积分对比**：显式欧拉（易爆炸）、半隐式欧拉（稳定，工业默认）、隐式欧拉（衰减更快）。
- **GPU 并行 + Kernel 融合**：每个积分器把”受力计算“与”位置/速度更新“合并在 **同一个 `@ti.kernel`** 内，最小化每帧的 Kernel 启动次数。
- **隐式 Kernel 单次启动**：用 `ti.static(range(N))` 把定点迭代在 **编译期** 展开成 N 个连续的并行 sub-pass，单帧只触发 1 次 Kernel launch。
- **`@ti.func` 强制内联**：`compute_forces_on()` 与 `clamp_velocity()` 全部声明为 `@ti.func`，由编译器内联到 Kernel 中以消除函数调用开销。
- **防爆速度钳制**：`MAX_VELOCITY` 上限保证显式欧拉在临界 dt 附近也不会立刻 NaN，避免直接崩溃。

## 2. 效果展示 

| 改变 k_s (弹簧系数) | 改变 k_d (阻尼系数) |
| :---: | :---: |
| ![改变 k_s](gif/work6_change_ks.gif) | ![改变 k_d](gif/work6_change_kd.gif) |
| *k_d 不变，实时 k_s 大小调节、三种积分方法对比* | *k_s 不变，实时 k_d 大小调节、三种积分方法对比* |

| 极端值 k_s | 极端值 k_d |
| :---: | :---: |
| ![极端 k_s](gif/work6_damping=0.gif) | ![极端 k_d](gif/work6_stiffness=2500.gif) |
| *k_d 为默认值，k_s 为 0* | *k_s 为默认值，k_d 为 2500* |

## 3. 项目架构

```text
Work6/
├── __init__.py        # 在子模块加载前先 ti.init(arch=ti.gpu)
├── config.py          # 布料 / 物理 / 时间步 / UI / 渲染 等所有常量
├── physics.py         # 任务1 字段定义  + 任务2 @ti.func 助手
├── solver.py          # 任务3 三个 @ti.kernel 积分求解器
├── main.py            # 任务1 初始化 Kernel + 任务4 GGUI 控制面板与渲染主循环
├── optional.py        # 选做实验：弹簧拓扑开关 + 球体碰撞 (自包含可独立运行)
├── gif/               # 项目效果展示 gif 图片
└── README.md          # 项目说明文档
```

| 文件 | 职责 |
|------|------|
| `__init__.py` | 在包被首次 `import` 时执行 `ti.init(arch=ti.gpu)`，保证后续子模块定义字段 / Kernel 时运行时已就绪。 |
| `config.py`   | 定义全部常量和相关参数 |
| `physics.py`  | 仿真状态字段（位置 `x`、速度 `v`、临时缓存 `new_x/new_v`、固定点掩码 `pin_mask`） + 力学 `@ti.func`：`clamp_velocity` 速度钳制、`compute_forces_on` 累加重力 / 阻尼 / 多类型弹簧力。 |
| `solver.py`   | 三种积分器 `@ti.kernel`：`step_explicit`、`step_semi_implicit`、`step_implicit_iter`；以及 Python 端 `step()` 调度入口。 |
| `main.py`     | 多个初始化 Kernel；每帧顶点更新；`ti.ui.Window` 三维场景、控制面板按钮与物理参数滑动条；参数变化自动重置；主循环。 |
| `optional.py` | **选做实验**：自包含的独立模块。Structural / Shear / Bending 可运行时开关 + 球体碰撞约束 (位置投影 + 法向速度清除)。仅使用半隐式欧拉。 |

## 4. 实现功能

### 4.1 场景初始化

**架构要求**：拆分为多个 `@ti.kernel`，在 Python 端按序调用。

| Kernel | 职责 |
|--------|------|
| `init_positions()` | 把 N×N 个质点放到 `(CLOTH_OFFSET.x + i*QUAD_SIZE, CLOTH_OFFSET.y, CLOTH_OFFSET.z + j*QUAD_SIZE)`，速度清零。 |
| `init_pin_mask_zero()` | 把 `pin_mask` 全部清零。 |
| `set_pin_points()` (Python) | 按 `PIN_POINTS`（四个角）配置写入 1。 |
| `init_triangle_indices()` | 为每个网格 cell `(i, j)` 生成 2 个三角形 (`v00,v10,v11`) 和 (`v00,v11,v01`)，供 `scene.mesh()` 渲染。 |
| `init_spring_line_indices()` | 为水平 + 垂直结构弹簧生成线段索引，供 `scene.lines()` 渲染线框。 |

GPU 同步由 Taichi 自动处理：连续的 `@ti.kernel` 调用之间有隐式 barrier，前一个 Kernel
完成所有 SIMT 线程后才会启动下一个。

### 4.2 力学计算与防爆处理 (`@ti.func`)

```python
@ti.func
def clamp_velocity(vel):
    """限制 |v| ≤ MAX_VELOCITY，按比例缩放。"""

@ti.func
def compute_forces_on(i, j, vel_field: ti.template(),
                       stiffness: ti.f32, damping: ti.f32) -> ti.math.vec3:
    """合力 = m*g + (-k_d * v) + Σ k_s * (|Δx| - l) * (Δx / |Δx|)。
       vel_field 通过 ti.template() 接受不同的速度场。"""
```

- **重力**：`F_g = m * g`。
- **全局阻尼**：`F_d = -k_d * v`。是否使用未来速度 `v_{t+1}` 决定了显式 vs 隐式。
- **弹簧力**：
  $$
  \mathbf{F}_a = -k_s\,(|\mathbf{x}_a - \mathbf{x}_b| - l)\,
  \frac{\mathbf{x}_a - \mathbf{x}_b}{|\mathbf{x}_a - \mathbf{x}_b|}
  $$
  对每个质点 (i, j)，遍历 12 个邻居偏移 (结构 4 + 剪切 4 + 弯曲 4)，
  累加每个邻居贡献的弹力。
- **防爆**：在每个积分器更新完 `new_v` 之后立即调用，把超过
  `MAX_VELOCITY` 的速度按比例缩到上限。这能避免显式欧拉在大 dt 或大 k_s 时
  瞬间发散为 NaN，使“接近临界 dt”的发散表现可视化。

`@ti.func` 在编译期被强制 inline 到调用它的 Kernel 中。这避免了 GPU 上昂贵的函数调用语义，同时让编译器看到所有上下文以做更激进的寄存器分配与公共子表达式消除。

> **关于 `ti.atomic_add`**：避免多线程
> 写入冲突，本实现采用 **per-particle 并行**（每个 (i, j) 线程对自己负责的
> 质点累加力到本地变量 `f`），不存在跨线程写冲突，因此天然无需原子加。
> 若采用 per-spring 并行（每条弹簧同时写入两端质点的力字段），则必须使用
> `ti.atomic_add` 才能保证正确性。

### 4.3 积分求解器实现

**架构要求**：受力计算与位置/速度更新合并在 **同一个 Kernel** 内，
最小化每帧循环的 Kernel 启动次数。

#### (a) 显式欧拉 `step_explicit`

$$
x_{t+1} = x_t + v_t \Delta t \,,\quad v_{t+1} = v_t + a_t \Delta t
$$

```python
# 用 v_t 同时算 new_x 和 new_v
for i, j in x:
    f = compute_forces_on(i, j, v, stiffness, damping)
    a = f / m
    new_x[i, j] = x[i, j] + v[i, j] * dt              # 用旧 v
    new_v[i, j] = clamp_velocity(v[i, j] + a * dt)    # 防爆
# 写回
for i, j in x:
    x[i, j], v[i, j] = new_x[i, j], new_v[i, j]
```

**特点**：能量持续累积。当 dt 较大或 k_s 较高时数值发散，布料表现为剧烈
抖动 + 速度被钳到上限 (v_max = MAX_VELOCITY)。

#### (b) 半隐式 / 辛欧拉 `step_semi_implicit`

$$
v_{t+1} = v_t + a_t \Delta t \,,\quad x_{t+1} = x_t + v_{\color{red}{t+1}} \Delta t
$$

先更新速度，再用 **新速度** 更新位置。在保留能量守恒的辛形式中，远远更稳定，
是物理引擎的工业默认。

#### (c) 隐式 / 反向欧拉 `step_implicit_iter`（定点迭代）

$$
v_{t+1} = v_t + a(x_t, v_{\color{red}{t+1}}) \Delta t
$$

F 中的阻尼项依赖未知的 `v_{t+1}`。我们用 **Jacobi-like 定点迭代** 近似求解：
以 `v_t` 为初值，反复用当前估计的 `v_pred` 重新计算阻尼力，得到下一个 `v_pred`。

**单 Kernel 实现技巧**：用 `ti.static(range(IMPLICIT_ITERS))` 在 **编译期** 把
定点迭代展开成 IMPLICIT_ITERS 个连续的并行 sub-pass。Taichi 在每对 top-level
parallel-for 之间自动插入 barrier，整个隐式求解只需 1 次 Kernel launch：

```python
@ti.kernel
def step_implicit_iter(dt, stiffness, damping):
    # 阶段1: 初始化迭代起点  v_pred = v_t
    for i, j in v:
        new_v[i, j] = v[i, j]

    # 阶段2: 编译期展开 8 个 parallel-pass
    for _ in ti.static(range(IMPLICIT_ITERS)):
        for i, j in v:
            f = compute_forces_on(i, j, new_v, stiffness, damping)   
            new_v[i, j] = clamp_velocity(v[i, j] + (f / m) * dt)

    # 阶段3: 应用 v_{t+1} 算 x_{t+1}, 并把 new_v 写回 v
    for i, j in x:
        if pin_mask[i, j] == 0:
            x[i, j] += new_v[i, j] * dt
        v[i, j] = new_v[i, j]
```

**特点**：阻尼项每个迭代都基于更新后的速度重新求值，相当于把阻尼分摊
到多次衰减，宏观上对振动能量收敛速度比半隐式更快、更”黏”。

### 4.4 渲染与 GGUI 交互

使用 `ti.ui.Window` 构建 3D 场景，包含：

- 布料 **三角网格** (`scene.mesh`, two-sided 双面渲染)
- 结构弹簧 **线框** (`scene.lines`, 仅水平/垂直结构弹簧以保持视觉清晰)
- 布料 **质点小球** (`scene.particles`)
- 环境光 + 点光源
- 鼠标右键 + 拖拽 旋转视角 (`camera.track_user_inputs`)

控制面板（`window.GUI`）：

| 控件 | 类型 | 默认 | 功能 |
|------|------|------|------|
| `[*] Explicit Euler (Explosive)`     | Button | OFF       | 切换到显式欧拉                                   |
| `[*] Semi-Implicit Euler (Stable)`   | Button | **ON**    | 切换到半隐式欧拉（默认）                          |
| `[*] Implicit Euler (Damped)`        | Button | OFF       | 切换到隐式欧拉                                   |
| `Pause / Resume Simulation`          | Button | Running   | 暂停 / 恢复仿真                                 |
| `Damping (k_d)`                      | Slider | 0.2       | 全局阻尼系数 (0–5)                              |
| `Stiffness (k_s)`                    | Slider | 600       | 弹簧劲度系数 (200–2500)                         |

按钮文本前缀 `[*]` 表示当前激活的方法，`[ ]` 表示未激活。

**自动重置**：主循环每帧记录上一次的 `method / damping / stiffness` 三个值，与
当前值比较，只要切换按钮或拖动滑动条就立即调用 `reset_cloth()`，
把布料拉回 **初始水平静止态**。这样每次切换方法或调整参数都从同一个初始条件
出发，可以直接对比不同方法 / 参数下的演化过程，无需手动按重置按钮。

## 5. 实验原理回顾

### 5.1 质点-弹簧模型

布料被离散成 N×N 个质点。质点 (i, j) 受三类力：

1. **重力**：$\mathbf{F}_g = m\mathbf{g}$
2. **全局阻尼力**：$\mathbf{F}_d = -k_d \mathbf{v}$
3. **弹簧力 (胡克定律)**：对每个邻居 (i', j') 弹簧力 (作用在 (i,j) 上)
   $$
   \mathbf{F}_s^{(i,j) \leftarrow (i',j')} = k_s \left( |\Delta\mathbf{x}| - l \right) \frac{\Delta\mathbf{x}}{|\Delta\mathbf{x}|},\quad \Delta\mathbf{x} = \mathbf{x}_{i',j'} - \mathbf{x}_{i,j}
   $$

合力对 12 个邻居（结构 4 + 剪切 4 + 弯曲 4）求和。

### 5.2 数值积分的稳定性差异

由牛顿第二定律 $\mathbf{a} = \mathbf{F}/m$，在离散时间步 $\Delta t$ 内：

| 方法 | 速度更新 | 位置更新 | 行为 |
|------|---------|---------|------|
| 显式欧拉 | $v_{t+1} = v_t + a(x_t, v_t)\Delta t$ | $x_{t+1} = x_t + v_t\Delta t$ | 能量持续增长 → 爆炸 |
| 半隐式 / 辛欧拉 | $v_{t+1} = v_t + a(x_t, v_t)\Delta t$ | $x_{t+1} = x_t + v_{t+1}\Delta t$ | 能量近似守恒 → 稳定 |
| 隐式 / 反向欧拉 | $v_{t+1} = v_t + a(x_t, v_{t+1})\Delta t$ | $x_{t+1} = x_t + v_{t+1}\Delta t$ | 能量持续耗散 → 衰减更快 |

显式欧拉之所以发散：胡克弹力 $\mathbf{F}_s \propto -k_s \mathbf{x}$ 本质是一个二阶
线性振子。显式欧拉对这个振子的离散映射的特征值 $|\lambda| = \sqrt{1 + (\omega \Delta t)^2} > 1$，
任何振动都会以 $\lambda^n$ 指数放大。半隐式 / 辛欧拉的特征值正好落在 **单位圆上**
（在适度 $\Delta t$ 下），保证能量长时间近似守恒。隐式欧拉的特征值 $|\lambda| < 1$，
振动逐渐衰减，但同时带来额外阻尼。

## 6. 运行方式

### 运行环境与启动

在项目根目录（`CG-lab`）下执行：

```bash
uv run -m src.Work6.main
```

### 交互操作说明

- **鼠标右键 + 拖拽**：旋转视角。
- **Control Panel**：
  1. 点击 `[ ] Explicit/Semi-Implicit/Implicit Euler` 切换三种积分方法（前缀
     `[*]` 表示当前激活）。**每次切换都会自动 reset 布料**。
  2. `Pause Simulation` / `Resume Simulation` 暂停或恢复。
  3. 拖动 `Damping (k_d)` 和 `Stiffness (k_s)` 滑动条调整物理参数。
     **每次调整都会自动 reset 布料**，方便和上一次参数的演化做对比。

## 7. 选做实验

选做部分实现两个扩展，全部位于独立模块 `optional.py` —— 自带仿真字段、力学计算
与渲染主循环，**不依赖** `physics.py` / `solver.py`，避免污染任务 1-4 主链路。

### 7.1 完善弹簧模型（Structural + Shear + Bending）

布料的质感很大程度上由弹簧拓扑决定。本模块将 12 邻居拆为 3 个语义分组，
对应抵抗布料的不同形变模式：

| 类型 | 偏移 (di, dj) | 邻居数 | 抵抗的形变 | 关闭时表现 |
|------|---------------|--------|------------|------------|
| **Structural** (始终开) | `(1,0) (0,1)` 及反向 | 4 | 拉伸 / 压缩 | — |
| **Shear** (可关) | `(1,1) (1,-1)` 及反向 | 4 | 剪切（对角变形） | 网格易出现平行四边形塌陷、对角扭曲 |
| **Bending** (可关) | `(2,0) (0,2)` 及反向 | 4 | 弯曲（小尺度褶皱） | 表面出现明显折叠、锯齿、高频褶皱 |

实现要点：

- 三类偏移在 Python 端分组定义并附带 `ALL_TYPES` 类型标记（0/1/2）。
- 在 `@ti.func compute_forces` 内，用 `ti.static(ALL_TYPES[k] == 1)` /
  `ti.static(ALL_TYPES[k] == 2)` 把挑选对应开关的判断 **编译期** 解析掉，
  GPU 上只剩一个 `if enabled == 1:` 的运行时分支 —— 几乎零开销。
- `enable_shear`、`enable_bending` 作为 `ti.i32` Kernel 参数传入，可在 UI 中
  实时切换；任意一个发生变化时自动 `reset_cloth()`，方便对比同初始条件下的演化。

实现效果：

| 弹簧消融 (All → -Bending → -Shear) | Bending vs Shear 对比 |
| :---: | :---: |
| ![弹簧消融](gif/work6_optional_spring_ablation.gif) | ![弹簧对比](gif/work6_optional_shear_vs_bending.gif) |
| *依次关闭 Bending、Shear，观察褶皱与剪切塌陷* | *单独缺失某一类弹簧时的形态差异* |

### 7.2 空间碰撞球体

每个物理子步在位置更新后追加一次 **球体投影约束**：

$$
\text{if } \|\mathbf{x} - \mathbf{c}\| < r:\quad
\mathbf{x} \leftarrow \mathbf{c} + \frac{\mathbf{x} - \mathbf{c}}{\|\mathbf{x} - \mathbf{c}\|}\,r,\quad
\mathbf{v} \leftarrow \mathbf{v} - \max(0,\,\mathbf{v}\cdot\mathbf{n})\,\mathbf{n}
$$

其中 $\mathbf{n} = (\mathbf{x}-\mathbf{c})/\|\mathbf{x}-\mathbf{c}\|$ 是球面朝外的单位法向。

实现要点：

- 球心 / 半径用 0-维 `ti.Vector.field` / `ti.field` 存储，UI 滑动条更新后
  Python 端单次写入即可被所有线程读取。
- 投影约束封装在 `@ti.func resolve_sphere_collision`，与位置更新融合在 **同一个 Kernel** 内 —— 无额外 Kernel launch 开销。
- 调节球体位置不触发 reset，可以实时"扫"球体看布料反应。

实现效果：

| 桌布罩球 (4 角固定) | 球体动态扫掠 (无固定) |
| :---: | :---: |
| ![桌布罩球](gif/work6_optional_sphere_drape.gif) | ![球体扫掠](gif/work6_optional_sphere_sweep.gif) |
| *四角悬挂，布料松弛贴合到球面* | *关闭 Pin，实时拖动 Sphere Y / R 观察布料反应* |

### 7.3 GUI 控制面板

| 控件 | 类型 | 默认 | 功能 |
|------|------|------|------|
| `[*] Shear Springs`     | Button | ON       | 切换剪切弹簧启用                            |
| `[*] Bending Springs`   | Button | ON       | 切换弯曲弹簧启用                            |
| `[*] Pin 4 Corners`     | Button | ON       | 切换四角固定（关闭则布料整体下落）           |
| `Pause / Resume`        | Button | Running  | 暂停 / 恢复                                |
| `Reset Cloth`           | Button | —        | 手动重置布料                                |
| `Damping (k_d)`         | Slider | 0.2      | 全局阻尼 (0–5)                              |
| `Stiffness (k_s)`       | Slider | 600      | 弹簧劲度 (200–2500)                         |
| `Sphere Y`              | Slider | 0.20     | 球心 Y 高度 (-0.10–0.45)                    |
| `Sphere Radius`         | Slider | 0.18     | 球半径 (0.05–0.30)                          |

自动重置触发条件：`enable_shear / enable_bending / use_pins / k_d / k_s`
任一变化即重置；**球体位置变化不重置**（允许扫描球体看实时反应）。

### 7.4 运行方式

在项目根目录（`CG-lab`）下执行：

```bash
uv run -m src.Work6.optional
```
