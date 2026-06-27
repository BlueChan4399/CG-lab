import taichi as ti


@ti.func
def ray_sphere_intersect(orig, dir, center, radius):
    """射线-球求交（解二次方程）。返回 (hit, t)。"""
    oc = orig - center
    a = dir.dot(dir)
    b = 2.0 * oc.dot(dir)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4.0 * a * c
    hit = False
    t = 1e20
    if discriminant >= 0.0:
        sqrt_disc = ti.math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        if t1 > 1e-4:
            t = t1
            hit = True
        elif t2 > 1e-4:
            t = t2
            hit = True
    return hit, t


@ti.func
def ray_plane_intersect(orig, dir, plane_y):
    """射线-水平无限大平面求交 (y = plane_y)。"""
    hit = False
    t = 1e20
    if ti.abs(dir.y) > 1e-6:
        t_temp = (plane_y - orig.y) / dir.y
        if t_temp > 1e-4:
            t = t_temp
            hit = True
    return hit, t


@ti.func
def reflect(I, N):
    """理想镜面反射方向公式 R = I - 2(I·N)N。"""
    return (I - 2.0 * I.dot(N) * N).normalized()


@ti.func
def refract(I, N, eta):
    """选做：Snell's Law 折射方向。
    参数:
        I    - 入射方向（单位向量，朝向表面）
        N    - 表面外法线（指向 I 的来源一侧）
        eta  - n_in / n_out （从入射介质到出射介质的相对折射率）
    返回:
        (refr_dir, tir)
        refr_dir - 折射方向（仅在 tir=False 时有效）
        tir      - 是否发生全反射（Total Internal Reflection）
    """
    cos_i = -I.dot(N)
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    refr = ti.math.vec3(0.0, 0.0, 0.0)
    tir = False
    if k < 0.0:
        tir = True
    else:
        refr = (eta * I + (eta * cos_i - ti.math.sqrt(k)) * N).normalized()
    return refr, tir


@ti.func
def checker_color(p, color_a, color_b):
    """通过交点 x、z 坐标的奇偶性判断棋盘格颜色。"""
    cx = ti.floor(p.x)
    cz = ti.floor(p.z)
    parity = (ti.cast(cx, ti.i32) + ti.cast(cz, ti.i32)) % 2
    out = color_a
    if parity != 0:
        out = color_b
    return out