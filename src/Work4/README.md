# CG-Lab4 - Whitted-Style Ray Tracing

## 1. 项目简介

本实验在 Work3 局部光照模型的基础上，进一步实现了经典的 **Whitted-Style 光线追踪**。
脱离 CPU 递归思维，使用 GPU 友好的 `for` 循环 + 吞吐量 (`throughput`) / 累加颜色 (`final_color`)
方案，在 Taichi Kernel 内完成多次反弹，渲染出包含**硬阴影**、**理想镜面反射**、**玻璃折射**与
**抗锯齿**的物理真实场景。

核心亮点：

- **迭代式光线弹射**：将传统递归算法改写为单 Kernel 内的 `for` 循环（最多 8 次），完美适配 GPU SIMT 模型。
- **多材质混合场景**：黑白棋盘格地面（漫反射）、红色漫反射球、银色纯镜面球、**玻璃球（折射 + TIR）** 同框出现。
- **Shadow Ray 硬阴影**：从交点向光源发射暗影射线，被遮挡时仅保留环境光分量。
- **数值精度修正**：沿法线方向偏移 `1e-4`，根除自相交导致的 Shadow Acne 黑色噪点。
- **选做 1 — Snell 折射玻璃球**：根据折射率 `1.5` 与 Snell's Law 计算透射方向，全反射 (TIR) 时回退为镜面反射。
- **选做 2 — MSAA 抗锯齿**：每像素发射 N 条带亚像素抖动的主射线后取均值，消除物体边缘锯齿。
- **实时 UI 交互**：通过 `ti.ui.Window` 滑动条动态调节光源三维坐标 (Light X/Y/Z)、最大弹射次数与 AA 采样数。

## 2. 效果展示

![必做实验效果](cg_Work4_demo1.gif)
*前景中央透明球为玻璃材质，可看到地面棋盘格透过球面被反向折射；左红球漫反射、右银球镜中世界、地面硬阴影。*

![选做实验效果](cg_Work4_demo2.gif)
*新增玻璃材质的红球。根据折射率计算透射光线方向，并在内部发生全反射时进行处理；实现抗锯齿 (Anti-Aliasing, MSAA)。*
## 3. 项目架构

```text
Work4/
├── __init__.py         # 模块初始化文件
├── config.py           # 场景配置（分辨率、相机、光源默认值、几何参数、材质、UI 范围）
├── raytracer.py        # 底层算法（射线-球/平面求交、反射、折射、棋盘格颜色采样）
├── main.py             # 主程序：场景求交调度、迭代弹射 Kernel、Phong+阴影着色、MSAA、UI 面板
└── README.md           # 项目说明文档
```

| 文件 | 职责 |
|------|------|
| `config.py`    | 窗口分辨率 `800×600`，相机/光源/物体几何与颜色，材质系数 `Ka/Kd/Ks/N`，镜面反射率 `0.8`，玻璃 IOR `1.5`，自相交偏移 `EPSILON=1e-4`，UI 范围。 |
| `raytracer.py` | `ray_sphere_intersect`、`ray_plane_intersect`、`reflect`、`refract`（Snell + TIR 标志）、`checker_color` 等纯计算 `@ti.func`。 |
| `main.py`      | `intersect_scene`/`get_normal`/`shade_diffuse`/`trace_ray`，MSAA 多采样 `render` Kernel，以及 `ti.ui.Window` 控件构建。 |

## 4. 实现功能（必做 + 选做）

### 任务 1：搭建包含平面的三维场景

在 `config.py` 中以**隐式方程**定义几何体，并通过 `hit_obj_id` 区分材质：

| ID | 物体 | 几何 | 材质 |
|:--:|------|------|------|
| 1 | **无限大平面 (Ground)** | `y = -1.0`，法线 `(0,1,0)` | 漫反射 + 黑白棋盘格（按交点 `floor(x)+floor(z)` 奇偶判定） |
| 2 | **红色漫反射球 (Red)**   | 圆心 `(-1.5, 0.0, 0.0)`，半径 `1.0` | 漫反射 |
| 3 | **银色镜面球 (Mirror)**  | 圆心 `(1.5, 0.0, 0.0)`，半径 `1.0`  | 纯镜面反射，反射率 `0.8` |
| 4 | **玻璃球 (Glass)** *(选做)* | 圆心 `(0.0, -0.4, 2.2)`，半径 `0.6` | 折射 + TIR，IOR `1.5`，微蓝染色 |

### 任务 2：实现基于迭代的光线弹射

GPU 不擅长递归，因此在 `main.py::trace_ray` 的每个像素中改写为如下 `for` 循环：

```python
throughput = vec3(1.0)
final_color = vec3(0.0)
for k in range(max_bounces):
    hit_obj, t = intersect_scene(ray_orig, ray_dir)
    if hit_obj == 0:                                  # miss → 天空
        final_color += throughput * BG_COLOR
        break
    p = ray_orig + t * ray_dir
    n = get_normal(hit_obj, p)
    if hit_obj == 3:                                  # 镜面 → 继续弹射
        ray_orig = p + n * EPSILON
        ray_dir  = reflect(ray_dir, n)
        throughput *= MIRROR_REFLECTIVITY             # 0.8
    elif hit_obj == 4:                                # 玻璃 → Snell 折射 / TIR
        ...                                           # 见任务 5
        throughput *= GLASS_TINT
    else:                                             # 漫反射 → 终止
        shade = shade_diffuse(p, n, view_dir, albedo, light_pos)
        final_color += throughput * shade
        break
```

- **吞吐 `throughput`**：累计每次镜面反射 / 玻璃透射的衰减系数，再乘到最终着色上。
- **累加 `final_color`**：仅在击中漫反射物体或天空时一次性写入，整个循环天然避免了递归。

### 任务 3：实现硬阴影与解决浮点数精度 Bug

漫反射着色函数 `shade_diffuse` 在计算 Phong 之前向光源发射一条暗影射线：

```python
shadow_orig = p + n * EPSILON                         # ★ 沿法线偏移 1e-4
hit_s, t_s = intersect_scene(shadow_orig, light_dir)
in_shadow  = (hit_s != 0) and (t_s < light_dist)
```

> **核心避坑点**：若直接以 `p` 作为暗影射线起点，由于浮点误差，射线会与自身表面立刻相交，
> 导致整个受光面被误判为阴影，渲染结果是满屏黑色噪点。**沿法线方向外偏移一个极小值**
> （公式 $\mathbf{P}_{new} = \mathbf{P} + \mathbf{N} \cdot \epsilon$）即可彻底根除。
> 同样的偏移也用于镜面反弹与折射时的新光线起点，避免无限自相交。

### 任务 4：完成 UI 交互面板

在 `main.py::run` 中使用 `ti.ui.Window` 构建实时控制面板（已加宽至 0.42×0.34，容纳完整文字）：

| UI 控件 | 范围 / 默认 | 功能 |
|---------|-------------|------|
| **Light X 滑动条** | `-8.0 ~ 8.0` / 默认 `1.5` | 动态改变点光源 X 坐标，观察阴影实时移动。 |
| **Light Y 滑动条** | `0.5 ~ 10.0` / 默认 `4.0` | 动态改变点光源 Y 坐标（高度）。 |
| **Light Z 滑动条** | `-8.0 ~ 8.0` / 默认 `2.0` | 动态改变点光源 Z 坐标，控制阴影前后投射方向。 |
| **Max Bounces 滑动条** | `1 ~ 8` / 默认 `4` | 最大弹射次数：`1` 时镜面/玻璃球变成纯黑暗球；`>=4` 时玻璃球完整折射 + 后方反射。 |
| **AA Samples 滑动条** | `1 ~ 8` / 默认 `1` | 选做 MSAA 每像素采样数：`1` 关闭抗锯齿，`8` 物体边缘最为平滑。 |

### 任务 5（选做 1）：Snell's Law 折射与玻璃材质

在 `raytracer.py` 中新增 `refract(I, N, eta)`：根据 Snell's Law 求解折射方向，并返回是否发生全反射：

$$\eta = \frac{n_{in}}{n_{out}}, \quad \cos\theta_t = \sqrt{1 - \eta^2(1 - \cos^2\theta_i)}$$

$$\mathbf{T} = \eta \mathbf{I} + (\eta\cos\theta_i - \cos\theta_t)\mathbf{N}$$

当 $1 - \eta^2(1-\cos^2\theta_i) < 0$ 时发生 **全反射 (Total Internal Reflection, TIR)**，回退为镜面反射。

在 `trace_ray` 内的玻璃球分支：

```python
front_face = ray_dir.dot(n) < 0.0
outward_n = n if front_face else -n
eta = (1.0 / GLASS_IOR) if front_face else GLASS_IOR
refr_dir, tir = refract(ray_dir, outward_n, eta)
if tir:                                                  # 全反射
    ray_orig = p + outward_n * EPSILON
    ray_dir = reflect(ray_dir, outward_n)
else:                                                    # 透射
    ray_orig = p - outward_n * EPSILON                   # 反向偏移到另一侧介质
    ray_dir = refr_dir
throughput *= GLASS_TINT
```

> **进出介质的处理**：通过 `front_face` 判断光线是从空气进入玻璃 (`eta = 1/1.5`) 还是从玻璃出来到空气 (`eta = 1.5`)，
> 并对应翻转法线方向 `outward_n`。折射时起点偏到法线**反方向**（进入新介质），全反射时起点偏到**同方向**（保留在原介质）。

### 任务 6（选做 2）：MSAA 抗锯齿

`render` Kernel 在像素级并行循环中嵌套一层 **样本循环**：

```python
for i, j in pixel_buffer:
    accum = vec3(0.0)
    for s in range(samples):
        jx = ti.random() if samples > 1 else 0.5         # 亚像素抖动
        jy = ti.random() if samples > 1 else 0.5
        u = (i + jx) / WINDOW_RES[0]
        v = (j + jy) / WINDOW_RES[1]
        ray_dir = ...                                    # 主射线方向
        accum += trace_ray(CAMERA_POS, ray_dir, max_bounces, light_pos)
    pixel_buffer[i, j] = clamp(accum / samples, 0, 1)
```

- `samples == 1` 走像素中心，无开销；`samples == N` 时每像素发射 N 条主射线并取均值。
- 由于循环嵌套在 `@ti.kernel` 内，仍保持 GPU 完全并行，性能近似线性下降。

## 5. 实验原理回顾

本实验采用经典的 **Whitted-Style** 光线追踪模型。当一条主光线 (Primary Ray) 从摄像机出发击中物体表面时：

1. **阴影测试**：从交点向光源方向发射一条暗影射线。如果该射线在到达光源前命中其他物体，则该点处于阴影中，仅保留环境光。
2. **材质分支**：
   - **漫反射 (Diffuse)**：按 Phong 模型计算颜色，并**终止该条光线的传播**。
   - **理想镜面 (Mirror)**：根据反射定律计算反射方向 $\mathbf{R} = \mathbf{L}_{in} - 2(\mathbf{L}_{in} \cdot \mathbf{N})\mathbf{N}$，生成新的反射光线继续传播。
   - **玻璃 (Glass，选做)**：根据 Snell 定律计算折射方向，全反射时退化为镜面反射。

## 6. 运行方式

### 运行环境与启动

确保环境中已安装 `taichi >= 1.7.4`。在项目根目录（`CG-lab`）下执行：

```bash
uv run -m src.Work4.main
```

### 交互操作说明

程序运行后，所有操作通过窗口右上角的 **Controls (UI 面板)** 用鼠标完成。

### 观察与测试建议

1. **观察镜中世界**：默认 `Max Bounces = 4`，右侧银球完整反射红球、地面与天空；正中玻璃球可见地面棋盘格被上下倒置（折射成像）。
2. **拉低弹射次数对比**：
   - `Max Bounces = 1` → 镜面球与玻璃球只命中"反射/折射方向"且无后续弹射，呈纯背景色。
   - `Max Bounces = 2` → 玻璃球只折射进入但还未折射出来，球心呈深色。
   - `Max Bounces = 4 ~ 8` → 完整透过玻璃看见后方场景，弹射越多镜中世界越深（`throughput=0.8^k`）。
3. **拖动光源**：滑动 `Light X / Y / Z`，观察地面阴影沿光源方向**实时移动**。
4. **MSAA 对比**：把 `AA Samples` 从 `1` 拉到 `4` 或 `8`，观察玻璃球与红球边缘从锯齿状变为平滑过渡（注意帧率会下降到原来的 1/N）。
5. **TIR 现象**：玻璃球边缘可看到一圈较深区域，是因为入射角超过临界角发生了**全反射**（光线被困在玻璃内部）。