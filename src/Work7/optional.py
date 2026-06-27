import math
import taichi as ti
from .config import (
    WINDOW_RES, N, QUAD_SIZE, CLOTH_OFFSET, PIN_POINTS,
    DT, SUBSTEPS,
    SPRING_STIFFNESS, DAMPING_DEFAULT,
    PARTICLE_MASS, MAX_VELOCITY, GRAVITY,
    DAMPING_RANGE, STIFFNESS_RANGE,
    CAMERA_POS_INIT, CAMERA_LOOKAT, CAMERA_UP, CAMERA_FOV,
    BG_COLOR, PARTICLE_COLOR, PARTICLE_RADIUS,
    POINT_LIGHT_POS, POINT_LIGHT_COLOR, AMBIENT_COLOR,
)

STRUCTURAL_OFFSETS = ((1, 0), (0, 1))    # 水平 / 垂直邻居
SHEAR_OFFSETS      = ((1, 1), (1, -1))   # 对角邻居
BENDING_OFFSETS    = ((2, 0), (0, 2))    # 隔一邻居（沿主轴）

ALL_OFFSETS = STRUCTURAL_OFFSETS + SHEAR_OFFSETS + BENDING_OFFSETS
ALL_TYPES = (
    (0,) * len(STRUCTURAL_OFFSETS) +
    (1,) * len(SHEAR_OFFSETS) +
    (2,) * len(BENDING_OFFSETS)
)
NUM_OFFSETS = len(ALL_OFFSETS)
REST_LENGTHS = tuple(
    math.sqrt(di * di + dj * dj) * QUAD_SIZE
    for di, dj in ALL_OFFSETS
)


# 球体参数
SPHERE_CX_DEFAULT = 0.0
SPHERE_CY_DEFAULT = 0.20
SPHERE_CZ_DEFAULT = 0.0
SPHERE_R_DEFAULT  = 0.18
SPHERE_Y_RANGE    = (-0.10, 0.45)
SPHERE_R_RANGE    = (0.05, 0.30)
SPHERE_COLOR      = (0.85, 0.55, 0.20)

# Taichi 
x        = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))   # 质点位置
v        = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))   # 质点速度
new_v    = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))   # 半隐式中间缓冲
pin_mask = ti.field(dtype=ti.i32, shape=(N, N))             # 固定点掩码

# 球体状态（0-维标量场，Python 端写入）
sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=())
sphere_radius = ti.field(dtype=ti.f32, shape=())

# 渲染缓冲
NUM_VERTICES            = N * N
NUM_TRIANGLE_INDICES    = 2 * (N - 1) * (N - 1) * 3
NUM_SPRING_LINE_INDICES = 2 * (2 * N * (N - 1))            # 仅结构弹簧线框

vertices         = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
triangle_indices = ti.field(dtype=ti.i32, shape=NUM_TRIANGLE_INDICES)
spring_indices   = ti.field(dtype=ti.i32, shape=NUM_SPRING_LINE_INDICES)
sphere_render    = ti.Vector.field(3, dtype=ti.f32, shape=1)


# 力学计算 (@ti.func 编译期内联)
@ti.func
def clamp_velocity(vel):
    """限制速度上限，防止显式 / 数值不稳定时爆炸。"""
    speed = vel.norm()
    out = vel
    if speed > MAX_VELOCITY:
        out = vel * (MAX_VELOCITY / speed)
    return out


@ti.func
def compute_forces(i, j, vel_field: ti.template(),
                   stiffness: ti.f32, damping: ti.f32,
                   enable_shear: ti.i32, enable_bending: ti.i32) -> ti.math.vec3:
    """合力 = 重力 + 阻尼 + 三类弹簧力 (Shear/Bending 可关)。"""
    f = ti.math.vec3(0.0, 0.0, 0.0)

    # 重力
    f += PARTICLE_MASS * GRAVITY

    # 全局阻尼
    f += -damping * vel_field[i, j]

    # 弹簧力
    for k in ti.static(range(NUM_OFFSETS)):
        di   = ti.static(ALL_OFFSETS[k][0])
        dj   = ti.static(ALL_OFFSETS[k][1])
        rest = ti.static(REST_LENGTHS[k])

        # 编译期选择"是否启用"的开关
        enabled = 1
        if ti.static(ALL_TYPES[k] == 1):
            enabled = enable_shear
        if ti.static(ALL_TYPES[k] == 2):
            enabled = enable_bending

        if enabled == 1:
            # +offset 邻居
            if 0 <= i + di < N and 0 <= j + dj < N:
                diff = x[i + di, j + dj] - x[i, j]
                length = diff.norm() + 1e-8
                f += stiffness * (length - rest) * (diff / length)
            # -offset 邻居
            if 0 <= i - di < N and 0 <= j - dj < N:
                diff = x[i - di, j - dj] - x[i, j]
                length = diff.norm() + 1e-8
                f += stiffness * (length - rest) * (diff / length)
    return f


@ti.func
def resolve_sphere_collision(pos, vel):
    """球体碰撞投影：质点穿入球内时投影到球面 + 清除法向负速度。

    单边约束：只在 v·n < 0 (朝球心运动) 时移除法向分量，避免抓住已经
    在远离球的质点。切向分量保留 → 布料可以沿球面"滑"。
    """
    c = sphere_center[None]
    r = sphere_radius[None]
    new_pos = pos
    new_vel = vel
    delta = pos - c
    dist  = delta.norm()
    if dist < r:
        n = delta / (dist + 1e-8)              # 朝外单位法向
        new_pos = c + n * r                    # 推到球面
        vn = new_vel.dot(n)
        if vn < 0.0:                           # 仅清除指向球心的分量
            new_vel = new_vel - vn * n
    return new_pos, new_vel


# 积分求解：半隐式欧拉 + 球体碰撞（融合在一个 Kernel）
@ti.kernel
def step_with_collision(dt: ti.f32, stiffness: ti.f32, damping: ti.f32,
                        enable_shear: ti.i32, enable_bending: ti.i32):
    """半隐式欧拉：先更速度，再用新速度更位置，最后做球体投影约束。

        Pass-A:  v_{t+1} = v_t + a(x_t, v_t) * dt
        Pass-B:  x_{t+1} = x_t + v_{t+1} * dt   ; 然后球体约束投影
    """
    # v_t → v_{t+1}
    for i, j in x:
        if pin_mask[i, j] == 1:
            new_v[i, j] = ti.math.vec3(0.0, 0.0, 0.0)
        else:
            f = compute_forces(i, j, v, stiffness, damping,
                               enable_shear, enable_bending)
            a = f / PARTICLE_MASS
            new_v[i, j] = clamp_velocity(v[i, j] + a * dt)

    # x_t → x_{t+1} + 球体投影
    for i, j in x:
        if pin_mask[i, j] == 0:
            tentative = x[i, j] + new_v[i, j] * dt
            corrected_x, corrected_v = resolve_sphere_collision(tentative, new_v[i, j])
            x[i, j] = corrected_x
            v[i, j] = corrected_v
        else:
            v[i, j] = ti.math.vec3(0.0, 0.0, 0.0)


# 初始化 Kernel
@ti.kernel
def init_positions():
    for i, j in x:
        x[i, j] = ti.math.vec3(
            CLOTH_OFFSET[0] + i * QUAD_SIZE,
            CLOTH_OFFSET[1],
            CLOTH_OFFSET[2] + j * QUAD_SIZE,
        )
        v[i, j] = ti.math.vec3(0.0, 0.0, 0.0)


@ti.kernel
def init_pin_mask_zero():
    for i, j in pin_mask:
        pin_mask[i, j] = 0


@ti.kernel
def init_triangle_indices():
    for i, j in ti.ndrange(N - 1, N - 1):
        cell_id = i * (N - 1) + j
        v00 = i * N + j
        v01 = i * N + (j + 1)
        v10 = (i + 1) * N + j
        v11 = (i + 1) * N + (j + 1)
        triangle_indices[cell_id * 6 + 0] = v00
        triangle_indices[cell_id * 6 + 1] = v10
        triangle_indices[cell_id * 6 + 2] = v11
        triangle_indices[cell_id * 6 + 3] = v00
        triangle_indices[cell_id * 6 + 4] = v11
        triangle_indices[cell_id * 6 + 5] = v01


@ti.kernel
def init_spring_line_indices():
    for i, j in ti.ndrange(N - 1, N):
        idx = i * N + j
        spring_indices[idx * 2 + 0] = i * N + j
        spring_indices[idx * 2 + 1] = (i + 1) * N + j
    offset_lines = (N - 1) * N
    for i, j in ti.ndrange(N, N - 1):
        idx = i * (N - 1) + j
        spring_indices[(offset_lines + idx) * 2 + 0] = i * N + j
        spring_indices[(offset_lines + idx) * 2 + 1] = i * N + (j + 1)


@ti.kernel
def update_render_vertices():
    for i, j in x:
        vertices[i * N + j] = x[i, j]


def set_pin_points():
    for (i, j) in PIN_POINTS:
        pin_mask[i, j] = 1


def reset_cloth(use_pins: bool):
    """重置布料到初始水平静止状态。use_pins 控制是否钉住 4 个角。"""
    init_positions()
    init_pin_mask_zero()
    if use_pins:
        set_pin_points()


def update_sphere_state(cx: float, cy: float, cz: float, r: float):
    """Python 端把球体参数写入 GPU 字段，并更新渲染缓冲。"""
    sphere_center[None] = ti.math.vec3(cx, cy, cz)
    sphere_radius[None] = r
    sphere_render[0] = ti.math.vec3(cx, cy, cz)


# 主循环
def run():
    # 静态初始化
    use_pins = True
    reset_cloth(use_pins)
    init_triangle_indices()
    init_spring_line_indices()

    # 球体初始状态
    sphere_cx = SPHERE_CX_DEFAULT
    sphere_cy = SPHERE_CY_DEFAULT
    sphere_cz = SPHERE_CZ_DEFAULT
    sphere_r  = SPHERE_R_DEFAULT
    update_sphere_state(sphere_cx, sphere_cy, sphere_cz, sphere_r)

    # 窗口 / 摄像机
    window = ti.ui.Window(
        "Lab 6 Optional: Spring Topology + Sphere Collision",
        WINDOW_RES, vsync=True,
    )
    canvas = window.get_canvas()
    canvas.set_background_color(BG_COLOR)
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(*CAMERA_POS_INIT)
    camera.lookat(*CAMERA_LOOKAT)
    camera.up(*CAMERA_UP)
    camera.fov(CAMERA_FOV)
    gui = window.get_gui()

    # 仿真控制状态
    damping        = DAMPING_DEFAULT
    stiffness      = SPRING_STIFFNESS
    enable_shear   = 1
    enable_bending = 1
    paused         = False
    sub_dt         = DT / SUBSTEPS

    # 变化检测（用于自动重置）  
    prev_damping, prev_stiffness   = damping, stiffness
    prev_shear, prev_bending       = enable_shear, enable_bending
    prev_use_pins                  = use_pins

    while window.running:
        # 控制面板
        with gui.sub_window("Optional Lab Control", 0.02, 0.02, 0.32, 0.72):
            gui.text("Spring Topology:")
            gui.text("(Structural always ON)")
            if gui.button(("[*] " if enable_shear == 1 else "[ ] ")
                          + "Shear Springs"):
                enable_shear = 1 - enable_shear
            if gui.button(("[*] " if enable_bending == 1 else "[ ] ")
                          + "Bending Springs"):
                enable_bending = 1 - enable_bending

            gui.text("")
            gui.text("Cloth Setup:")
            if gui.button(("[*] " if use_pins else "[ ] ")
                          + "Pin 4 Corners"):
                use_pins = not use_pins
            if gui.button("Resume Simulation" if paused else "Pause Simulation"):
                paused = not paused
            if gui.button("Reset Cloth"):
                reset_cloth(use_pins)

            gui.text("")
            gui.text("Physics Parameters:")
            damping = gui.slider_float("Damping (k_d)",   damping,
                                       DAMPING_RANGE[0],   DAMPING_RANGE[1])
            stiffness = gui.slider_float("Stiffness (k_s)", stiffness,
                                         STIFFNESS_RANGE[0], STIFFNESS_RANGE[1])

            gui.text("")
            gui.text("Sphere Collider:")
            sphere_cy = gui.slider_float("Sphere Y",      sphere_cy,
                                         SPHERE_Y_RANGE[0], SPHERE_Y_RANGE[1])
            sphere_r  = gui.slider_float("Sphere Radius", sphere_r,
                                         SPHERE_R_RANGE[0], SPHERE_R_RANGE[1])

            gui.text("")
            gui.text(f"Status: {'PAUSED' if paused else 'RUNNING'}")
            gui.text(f"Shear:   {'ON ' if enable_shear else 'OFF'} | "
                     f"Bending: {'ON ' if enable_bending else 'OFF'}")
            gui.text(f"Pins:    {'ON ' if use_pins else 'OFF'}")
            gui.text("Tip: Right-click + drag to rotate view")

        # 每帧同步球体到 GPU
        update_sphere_state(sphere_cx, sphere_cy, sphere_cz, sphere_r)

        # 自动重置检测
        # 弹簧类型 / 物理参数 / 钉点设置变化 → 自动 reset
        # 球体位置变化不重置（允许用户动态扫描球体观察布料反应）
        if (damping != prev_damping or stiffness != prev_stiffness or
                enable_shear != prev_shear or enable_bending != prev_bending or
                use_pins != prev_use_pins):
            reset_cloth(use_pins)
        prev_damping, prev_stiffness = damping, stiffness
        prev_shear, prev_bending     = enable_shear, enable_bending
        prev_use_pins                = use_pins

        # 仿真推进
        if not paused:
            for _ in range(SUBSTEPS):
                step_with_collision(sub_dt, stiffness, damping,
                                    enable_shear, enable_bending)

        # 渲染
        update_render_vertices()

        camera.track_user_inputs(window, movement_speed=0.04,
                                 hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light(AMBIENT_COLOR)
        scene.point_light(pos=POINT_LIGHT_POS, color=POINT_LIGHT_COLOR)

        # 布料
        scene.mesh(vertices, indices=triangle_indices,
                   color=PARTICLE_COLOR, two_sided=True)
        scene.lines(vertices, indices=spring_indices,
                    width=1.5, color=(0.10, 0.30, 0.55))
        scene.particles(vertices, radius=PARTICLE_RADIUS, color=PARTICLE_COLOR)

        # 球体（单粒子 + 自定义大半径）
        scene.particles(sphere_render, radius=sphere_r, color=SPHERE_COLOR)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    run()
