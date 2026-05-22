# 三种数值积分求解器 (@ti.kernel)。

import taichi as ti
from .config import (
    PARTICLE_MASS,
    IMPLICIT_ITERS,
    METHOD_EXPLICIT, METHOD_SEMI_IMPLICIT, METHOD_IMPLICIT,
)
from .physics import (
    x, v, new_x, new_v, pin_mask,
    compute_forces_on, clamp_velocity,
)


# 方法 1：显式欧拉 (Explicit Euler) 
@ti.kernel
def step_explicit(dt: ti.f32, stiffness: ti.f32, damping: ti.f32):
    # 基于 t 时刻状态预测 t+1
    for i, j in x:
        if pin_mask[i, j] == 1:                               # 固定点不动
            new_x[i, j] = x[i, j]
            new_v[i, j] = ti.math.vec3(0.0, 0.0, 0.0)
        else:
            f = compute_forces_on(i, j, v, stiffness, damping)
            a = f / PARTICLE_MASS
            new_x[i, j] = x[i, j] + v[i, j] * dt              # 显式：用旧 v
            new_v[i, j] = clamp_velocity(v[i, j] + a * dt)    # 防爆钳制

    # 统一写回
    for i, j in x:
        x[i, j] = new_x[i, j]
        v[i, j] = new_v[i, j]

# 方法 2：半隐式 / 辛欧拉 (Semi-Implicit / Symplectic Euler)
@ti.kernel
def step_semi_implicit(dt: ti.f32, stiffness: ti.f32, damping: ti.f32):
    # 用 v_t 算 a_t → v_{t+1}
    for i, j in x:
        if pin_mask[i, j] == 1:
            new_v[i, j] = ti.math.vec3(0.0, 0.0, 0.0)
        else:
            f = compute_forces_on(i, j, v, stiffness, damping)
            a = f / PARTICLE_MASS
            new_v[i, j] = clamp_velocity(v[i, j] + a * dt)

    # 用 v_{t+1} 更新位置，并写回速度
    for i, j in x:
        if pin_mask[i, j] == 0:
            x[i, j] = x[i, j] + new_v[i, j] * dt
        v[i, j] = new_v[i, j]


# 方法 3：隐式 / 反向欧拉 (Implicit / Backward Euler) — 定点迭代
@ti.kernel
def step_implicit_iter(dt: ti.f32, stiffness: ti.f32, damping: ti.f32):
    # 阶段1：初始化迭代初值 v_pred = v_t
    for i, j in v:
        new_v[i, j] = v[i, j]

    # 阶段2：编译期展开 IMPLICIT_ITERS 个并行 sub-pass
    for _ in ti.static(range(IMPLICIT_ITERS)):
        for i, j in v:
            if pin_mask[i, j] == 1:
                new_v[i, j] = ti.math.vec3(0.0, 0.0, 0.0)
            else:
                f = compute_forces_on(i, j, new_v, stiffness, damping)
                a = f / PARTICLE_MASS
                new_v[i, j] = clamp_velocity(v[i, j] + a * dt)

    # 阶段3：用收敛后的 v_{t+1} 更新位置并写回速度
    for i, j in x:
        if pin_mask[i, j] == 0:
            x[i, j] = x[i, j] + new_v[i, j] * dt
        v[i, j] = new_v[i, j]

# 调度入口：根据方法编号分派
def step(method: int, dt: float, stiffness: float, damping: float):
    """Python 端调度：按 UI 选择的积分方法调用对应 Kernel。"""
    if method == METHOD_EXPLICIT:
        step_explicit(dt, stiffness, damping)
    elif method == METHOD_SEMI_IMPLICIT:
        step_semi_implicit(dt, stiffness, damping)
    else:
        step_implicit_iter(dt, stiffness, damping)
