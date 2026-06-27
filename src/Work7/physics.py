import math
import taichi as ti
from .config import (
    N, QUAD_SIZE, PARTICLE_MASS, MAX_VELOCITY, GRAVITY,
    SPRING_OFFSETS, NUM_SPRING_OFFSETS,
)

# 弹簧原长
REST_LENGTHS = tuple(
    math.sqrt(di * di + dj * dj) * QUAD_SIZE
    for di, dj in SPRING_OFFSETS
)

# 仿真状态字段 (Taichi Fields)
# 当前时刻状态
x = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))         # 质点位置
v = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))         # 质点速度

# 下一时刻 / 隐式定点迭代的中间缓冲
new_x = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
new_v = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))

# 固定点掩码：1=被钉住, 0=自由运动
pin_mask = ti.field(dtype=ti.i32, shape=(N, N))


# 力学计算与防爆处理 (@ti.func，编译期 inline)

@ti.func
def clamp_velocity(vel):
    """限制速度模长，防止显式欧拉等不稳定方法的数值爆炸。
    当 |v| > MAX_VELOCITY 时按比例缩放至上限，方向保留。
    """
    speed = vel.norm()
    out = vel
    if speed > MAX_VELOCITY:
        out = vel * (MAX_VELOCITY / speed)
    return out


@ti.func
def compute_forces_on(i, j, vel_field: ti.template(),
                       stiffness: ti.f32, damping: ti.f32) -> ti.math.vec3:
    """计算质点 (i, j) 上的合力 = 重力 + 阻尼力 + 弹簧力。"""
    f = ti.math.vec3(0.0, 0.0, 0.0)

    # 重力 F_g = m * g
    f += PARTICLE_MASS * GRAVITY

    # 全局阻尼力 F_d = -k_d * v
    f += -damping * vel_field[i, j]

    # 弹簧力 = 结构 + 剪切 + 弯曲
    # 每个对称偏移 (di, dj) 同时处理 +/- 两个方向上的邻居。
    # 边界检查后再读取 x，避免越界。
    for k in ti.static(range(NUM_SPRING_OFFSETS)):
        di = ti.static(SPRING_OFFSETS[k][0])
        dj = ti.static(SPRING_OFFSETS[k][1])
        rest = ti.static(REST_LENGTHS[k])
        # +offset 方向的邻居
        if 0 <= i + di < N and 0 <= j + dj < N:
            diff = x[i + di, j + dj] - x[i, j]
            length = diff.norm() + 1e-8
            f += stiffness * (length - rest) * (diff / length)
        # -offset 方向的邻居
        if 0 <= i - di < N and 0 <= j - dj < N:
            diff = x[i - di, j - dj] - x[i, j]
            length = diff.norm() + 1e-8
            f += stiffness * (length - rest) * (diff / length)
    return f

