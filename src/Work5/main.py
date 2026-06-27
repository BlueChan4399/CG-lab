import taichi as ti

ti.init(arch=ti.gpu)

from .config import *
from .raytracer import *


pixel_buffer = ti.Vector.field(3, dtype=ti.f32, shape=WINDOW_RES)


# ============================================================
#  场景求交：返回 (hit_obj_id, t)
#    1 = 平面, 2 = 红色漫反射球, 3 = 银色镜面球, 4 = 玻璃球, 0 = miss
# ============================================================
@ti.func
def intersect_scene(orig, dir):
    t_min = 1e20
    hit_obj = 0

    h1, t1 = ray_plane_intersect(orig, dir, PLANE_Y)
    if h1 and t1 < t_min:
        t_min = t1
        hit_obj = 1

    h2, t2 = ray_sphere_intersect(orig, dir, RED_SPHERE_CENTER, RED_SPHERE_RADIUS)
    if h2 and t2 < t_min:
        t_min = t2
        hit_obj = 2

    h3, t3 = ray_sphere_intersect(orig, dir, MIRROR_SPHERE_CENTER, MIRROR_SPHERE_RADIUS)
    if h3 and t3 < t_min:
        t_min = t3
        hit_obj = 3

    h4, t4 = ray_sphere_intersect(orig, dir, GLASS_SPHERE_CENTER, GLASS_SPHERE_RADIUS)
    if h4 and t4 < t_min:
        t_min = t4
        hit_obj = 4

    return hit_obj, t_min


@ti.func
def get_normal(hit_obj, p):
    n = ti.math.vec3(0.0, 1.0, 0.0)
    if hit_obj == 1:
        n = PLANE_NORMAL
    elif hit_obj == 2:
        n = (p - RED_SPHERE_CENTER).normalized()
    elif hit_obj == 3:
        n = (p - MIRROR_SPHERE_CENTER).normalized()
    elif hit_obj == 4:
        n = (p - GLASS_SPHERE_CENTER).normalized()
    return n


@ti.func
def get_diffuse_albedo(hit_obj, p):
    """漫反射物体（平面 / 红球）的本征颜色。"""
    c = ti.math.vec3(0.0)
    if hit_obj == 1:
        c = checker_color(p, PLANE_COLOR_A, PLANE_COLOR_B)
    elif hit_obj == 2:
        c = RED_SPHERE_COLOR
    return c


@ti.func
def shade_diffuse(p, n, view_dir, albedo, light_pos):
    """对漫反射物体执行 Phong 着色 + 硬阴影（Shadow Ray）。"""
    light_dir = (light_pos - p).normalized()
    light_dist = (light_pos - p).norm()

    # 任务3：核心避坑点 —— 沿法线偏移 EPSILON，避免自相交（Shadow Acne）
    shadow_orig = p + n * EPSILON
    hit_s, t_s = intersect_scene(shadow_orig, light_dir)
    in_shadow = (hit_s != 0) and (t_s < light_dist)

    ambient = KA * LIGHT_COLOR * albedo
    color = ambient
    if not in_shadow:
        diffuse = KD * LIGHT_COLOR * ti.max(0.0, n.dot(light_dir)) * albedo
        H = (light_dir + view_dir).normalized()
        specular = KS * LIGHT_COLOR * (ti.max(0.0, n.dot(H)) ** SHININESS)
        color = ambient + diffuse + specular
    return color


# ============================================================
#  单条光线追踪 (任务2 + 选做折射)
# ============================================================
@ti.func
def trace_ray(ray_orig, ray_dir, max_bounces, light_pos):
    throughput = ti.math.vec3(1.0, 1.0, 1.0)
    final_color = ti.math.vec3(0.0, 0.0, 0.0)
    terminated = False

    for k in range(max_bounces):
        hit_obj, t = intersect_scene(ray_orig, ray_dir)

        if hit_obj == 0:
            final_color += throughput * BG_COLOR
            terminated = True
            break

        p = ray_orig + t * ray_dir
        n = get_normal(hit_obj, p)

        if hit_obj == 3:
            # === 镜面物体：反射后继续弹射 ===
            ray_orig = p + n * EPSILON
            ray_dir = reflect(ray_dir, n)
            throughput *= MIRROR_REFLECTIVITY

        elif hit_obj == 4:
            # === 玻璃物体（选做）：Snell 折射 + TIR 全反射回退 ===
            front_face = ray_dir.dot(n) < 0.0
            outward_n = n
            eta = 1.0 / GLASS_IOR
            if not front_face:
                outward_n = -n
                eta = GLASS_IOR
            refr_dir, tir = refract(ray_dir, outward_n, eta)
            if tir:
                # 发生全反射 -> 走反射路径
                ray_orig = p + outward_n * EPSILON
                ray_dir = reflect(ray_dir, outward_n)
            else:
                # 折射穿过界面 -> 起点偏到另一侧
                ray_orig = p - outward_n * EPSILON
                ray_dir = refr_dir
            throughput *= GLASS_TINT

        else:
            # === 漫反射物体（平面 / 红球）：Phong 着色 + 阴影，结束 ===
            albedo = get_diffuse_albedo(hit_obj, p)
            view_dir = (-ray_dir).normalized()
            shade = shade_diffuse(p, n, view_dir, albedo, light_pos)
            final_color += throughput * shade
            terminated = True
            break

    if not terminated:
        # 弹射次数耗尽仍未命中漫反射物体 —— 视为指向天空
        final_color += throughput * BG_COLOR

    return final_color


# ============================================================
#  渲染核心：每像素 MSAA 多采样（选做） + 迭代弹射
# ============================================================
@ti.kernel
def render(light_x: ti.f32, light_y: ti.f32, light_z: ti.f32,
           max_bounces: ti.i32, samples: ti.i32):
    light_pos = ti.math.vec3(light_x, light_y, light_z)
    aspect = WINDOW_RES[0] / WINDOW_RES[1]
    tan_half_fov = ti.math.tan(ti.math.radians(FOV) / 2.0)

    for i, j in pixel_buffer:
        accum = ti.math.vec3(0.0, 0.0, 0.0)
        for s in range(samples):
            # samples == 1 时取像素中心，>1 时在像素内随机抖动（选做：MSAA）
            jx = 0.5
            jy = 0.5
            if samples > 1:
                jx = ti.random()
                jy = ti.random()
            u = (i + jx) / WINDOW_RES[0]
            v = (j + jy) / WINDOW_RES[1]
            ray_dir = ti.math.vec3(
                (2.0 * u - 1.0) * aspect * tan_half_fov,
                (2.0 * v - 1.0) * tan_half_fov,
                -1.0,
            ).normalized()
            accum += trace_ray(CAMERA_POS, ray_dir, max_bounces, light_pos)

        avg = accum / ti.f32(samples)
        pixel_buffer[i, j] = ti.math.clamp(avg, 0.0, 1.0)


# ============================================================
#  任务4：UI 交互窗口（已加宽以容纳完整文字）
# ============================================================
def run():
    window = ti.ui.Window("Lab 4: Whitted-Style Ray Tracing", WINDOW_RES)
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x = LIGHT_POS_DEFAULT[0]
    light_y = LIGHT_POS_DEFAULT[1]
    light_z = LIGHT_POS_DEFAULT[2]
    max_bounces = MAX_BOUNCES_DEFAULT
    samples = SAMPLES_DEFAULT

    while window.running:
        render(light_x, light_y, light_z, max_bounces, samples)
        canvas.set_image(pixel_buffer)

        # 加宽到 0.42（原 0.26），并把面板向左挪到 0.56，避免最后一行被截断
        with gui.sub_window("Controls", 0.56, 0.02, 0.42, 0.34):
            light_x = gui.slider_float("Light X", light_x, LIGHT_RANGE[0], LIGHT_RANGE[1])
            light_y = gui.slider_float("Light Y", light_y, LIGHT_Y_RANGE[0], LIGHT_Y_RANGE[1])
            light_z = gui.slider_float("Light Z", light_z, LIGHT_RANGE[0], LIGHT_RANGE[1])
            max_bounces = gui.slider_int(
                "Max Bounces", max_bounces,
                MAX_BOUNCES_RANGE[0], MAX_BOUNCES_RANGE[1],
            )
            samples = gui.slider_int(
                "AA Samples", samples,
                SAMPLES_RANGE[0], SAMPLES_RANGE[1],
            )
            gui.text(f"Bounces={max_bounces}  AA={samples}x  (1=off, 8=smoothest)")

        window.show()


if __name__ == "__main__":
    run()