import taichi as ti
from .config import (
    WINDOW_RES, N, QUAD_SIZE, CLOTH_OFFSET, PIN_POINTS,
    DT, SUBSTEPS,
    SPRING_STIFFNESS, DAMPING_DEFAULT,
    METHOD_EXPLICIT, METHOD_SEMI_IMPLICIT, METHOD_IMPLICIT, METHOD_NAMES,
    DAMPING_RANGE, STIFFNESS_RANGE,
    CAMERA_POS_INIT, CAMERA_LOOKAT, CAMERA_UP, CAMERA_FOV,
    BG_COLOR, PARTICLE_COLOR, PARTICLE_RADIUS,
    POINT_LIGHT_POS, POINT_LIGHT_COLOR, AMBIENT_COLOR,
)
from .physics import x, v, pin_mask
from .solver import step

# 渲染缓冲：把 (N,N) 网格展平为长度 N*N 的顶点数组
NUM_VERTICES = N * N
NUM_TRIANGLES = 2 * (N - 1) * (N - 1)
NUM_TRIANGLE_INDICES = NUM_TRIANGLES * 3
NUM_SPRING_LINES = 2 * N * (N - 1)               # 结构弹簧（水平+垂直）
NUM_SPRING_LINE_INDICES = NUM_SPRING_LINES * 2

vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
triangle_indices = ti.field(dtype=ti.i32, shape=NUM_TRIANGLE_INDICES)
spring_indices = ti.field(dtype=ti.i32, shape=NUM_SPRING_LINE_INDICES)

# 场景初始化 (拆分为多个 @ti.kernel ，按顺序在 Python 端调用)
# GPU 同步：每个 kernel 启动结束后, Taichi 会自动等待其完成再启动下一个

@ti.kernel
def init_positions():
    """初始化质点位置：构建一个 N×N 水平网格，速度清零。"""
    for i, j in x:
        x[i, j] = ti.math.vec3(
            CLOTH_OFFSET[0] + i * QUAD_SIZE,
            CLOTH_OFFSET[1],
            CLOTH_OFFSET[2] + j * QUAD_SIZE,
        )
        v[i, j] = ti.math.vec3(0.0, 0.0, 0.0)


@ti.kernel
def init_pin_mask_zero():
    """清空固定点掩码。"""
    for i, j in pin_mask:
        pin_mask[i, j] = 0


@ti.kernel
def init_triangle_indices():
    """初始化布料三角网格索引：每个 (i, j) 网格 cell 生成 2 个三角形。"""
    for i, j in ti.ndrange(N - 1, N - 1):
        cell_id = i * (N - 1) + j
        v00 = i * N + j
        v01 = i * N + (j + 1)
        v10 = (i + 1) * N + j
        v11 = (i + 1) * N + (j + 1)
        # ▲ 1: (v00, v10, v11)
        triangle_indices[cell_id * 6 + 0] = v00
        triangle_indices[cell_id * 6 + 1] = v10
        triangle_indices[cell_id * 6 + 2] = v11
        # ▲ 2: (v00, v11, v01)
        triangle_indices[cell_id * 6 + 3] = v00
        triangle_indices[cell_id * 6 + 4] = v11
        triangle_indices[cell_id * 6 + 5] = v01


@ti.kernel
def init_spring_line_indices():
    """初始化结构弹簧的线段索引（用于线框渲染，仅显示结构弹簧避免视觉拥挤）。"""
    # 水平弹簧 (i,j)-(i+1,j)：共 (N-1)*N 条
    for i, j in ti.ndrange(N - 1, N):
        idx = i * N + j
        spring_indices[idx * 2 + 0] = i * N + j
        spring_indices[idx * 2 + 1] = (i + 1) * N + j
    # 垂直弹簧 (i,j)-(i,j+1)：共 N*(N-1) 条
    offset_lines = (N - 1) * N
    for i, j in ti.ndrange(N, N - 1):
        idx = i * (N - 1) + j
        spring_indices[(offset_lines + idx) * 2 + 0] = i * N + j
        spring_indices[(offset_lines + idx) * 2 + 1] = i * N + (j + 1)


@ti.kernel
def update_render_vertices():
    """每帧将 (N,N) 形态的位置场拷贝到展平顶点缓冲。"""
    for i, j in x:
        vertices[i * N + j] = x[i, j]


def set_pin_points():
    """Python 端按 PIN_POINTS 配置设置固定点（标量写入 Taichi 字段）。"""
    for (i, j) in PIN_POINTS:
        pin_mask[i, j] = 1


def reset_cloth():
    """重置整块布料到初始静止状态。"""
    init_positions()
    init_pin_mask_zero()
    set_pin_points()

# GGUI 渲染循环 + 控制面板
def run():
    # 按序触发多个初始化 Kernel
    init_positions()
    init_pin_mask_zero()
    set_pin_points()
    init_triangle_indices()
    init_spring_line_indices()

    # 创建窗口与场景
    window = ti.ui.Window("Lab 6: Mass-Spring Cloth Simulation", WINDOW_RES, vsync=True)
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
    method = METHOD_SEMI_IMPLICIT     # 默认稳定方法
    damping = DAMPING_DEFAULT
    stiffness = SPRING_STIFFNESS
    paused = False
    sub_dt = DT / SUBSTEPS            # 子步时间步长

    # 上一帧的控制状态
    prev_method = method
    prev_damping = damping
    prev_stiffness = stiffness

    while window.running:
        # 控制面板   
        with gui.sub_window("Control Panel", 0.02, 0.02, 0.28, 0.50):
            gui.text("Integration Method:")
            if gui.button(("[*] " if method == METHOD_EXPLICIT else "[ ] ")
                          + "Explicit Euler (Explosive)"):
                method = METHOD_EXPLICIT
            if gui.button(("[*] " if method == METHOD_SEMI_IMPLICIT else "[ ] ")
                          + "Semi-Implicit Euler (Stable)"):
                method = METHOD_SEMI_IMPLICIT
            if gui.button(("[*] " if method == METHOD_IMPLICIT else "[ ] ")
                          + "Implicit Euler (Damped)"):
                method = METHOD_IMPLICIT

            gui.text("")
            # 暂停按钮
            if gui.button("Resume Simulation" if paused else "Pause Simulation"):
                paused = not paused

            gui.text("")
            gui.text("Physics Parameters:")
            damping = gui.slider_float("Damping (k_d)",   damping,
                                       DAMPING_RANGE[0],   DAMPING_RANGE[1])
            stiffness = gui.slider_float("Stiffness (k_s)", stiffness,
                                         STIFFNESS_RANGE[0], STIFFNESS_RANGE[1])

            gui.text("")
            gui.text(f"Status: {'PAUSED' if paused else 'RUNNING'}")
            gui.text(f"Method: {METHOD_NAMES[method]}")
            gui.text(f"Substep dt = {sub_dt:.5f} s")
            gui.text(f"Substeps/frame = {SUBSTEPS}")
            gui.text("Auto-reset on method/k_d/k_s change")
            gui.text("Tip: Right-click + drag to rotate view")

        # 自动重置检测
        # 切换积分方法 或 调整任一物理参数 → 自动 reset_cloth()
        if (method != prev_method or
                damping != prev_damping or
                stiffness != prev_stiffness):
            reset_cloth()
        prev_method = method
        prev_damping = damping
        prev_stiffness = stiffness

        # 仿真推进
        if not paused:
            for _ in range(SUBSTEPS):
                step(method, sub_dt, stiffness, damping)

        # 渲染
        update_render_vertices()

        camera.track_user_inputs(window, movement_speed=0.04,
                                  hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light(AMBIENT_COLOR)
        scene.point_light(pos=POINT_LIGHT_POS, color=POINT_LIGHT_COLOR)

        # 布料：三角网格
        scene.mesh(vertices, indices=triangle_indices,
                    color=PARTICLE_COLOR, two_sided=True)
        # 结构弹簧线框
        scene.lines(vertices, indices=spring_indices,
                     width=1.5, color=(0.10, 0.30, 0.55))
        # 质点小球
        scene.particles(vertices, radius=PARTICLE_RADIUS, color=PARTICLE_COLOR)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    run()
