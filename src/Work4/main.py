import math
import taichi as ti

ti.init(arch=ti.gpu)

from .config import *
from .raytracer import *

pixel_buffer = ti.Vector.field(3, dtype=ti.f32, shape=WINDOW_RES)

@ti.func
def intersect_scene(orig, dir):
    t_min = 1e20
    hit_obj = 0 
    h1, t1 = ray_sphere_intersect(orig, dir, SPHERE_CENTER, SPHERE_RADIUS)
    if h1 and t1 < t_min: t_min = t1; hit_obj = 1
    h2, t2 = ray_cone_intersect(orig, dir, CONE_APEX, CONE_BASE_Y, CONE_RADIUS)
    if h2 and t2 < t_min: t_min = t2; hit_obj = 2
    h3, t3 = ray_plane_intersect(orig, dir, PLANE_Y)
    if h3 and t3 < t_min: t_min = t3; hit_obj = 3
    return hit_obj, t_min

@ti.func
def get_normal(hit_obj, p):
    n = ti.math.vec3(0.0, 1.0, 0.0)
    if hit_obj == 1: n = (p - SPHERE_CENTER).normalized()
    elif hit_obj == 2: n = get_cone_normal(p, CONE_APEX, CONE_BASE_Y, CONE_RADIUS)
    elif hit_obj == 3: n = ti.math.vec3(0.0, 1.0, 0.0)
    return n

@ti.kernel
def render(ka: ti.f32, kd: ti.f32, ks: ti.f32, shininess: ti.f32, use_blinn: ti.i32):
    for i, j in pixel_buffer:
        u = (i + 0.5) / WINDOW_RES[0]
        v = (j + 0.5) / WINDOW_RES[1]
        aspect = WINDOW_RES[0] / WINDOW_RES[1]
        tan_half_fov = ti.math.tan(ti.math.radians(60.0) / 2.0)
        
        ray_dir = ti.math.vec3((2.0*u-1.0)*aspect*tan_half_fov, (2.0*v-1.0)*tan_half_fov, -1.0).normalized()
        hit_obj, t = intersect_scene(CAMERA_POS, ray_dir)
        
        color = BG_COLOR
        if hit_obj != 0:
            p = CAMERA_POS + t * ray_dir
            n = get_normal(hit_obj, p)
            
            obj_color = ti.math.vec3(0.0)
            if hit_obj == 1: obj_color = SPHERE_COLOR
            elif hit_obj == 2: obj_color = CONE_COLOR
            elif hit_obj == 3: obj_color = PLANE_COLOR

            light_dir = (LIGHT_POS - p).normalized()
            view_dir = (-ray_dir).normalized()
            
            # 阴影射线稍微偏移法线方向，避免自遮挡
            shadow_orig = p + n * 1e-3
            hit_s, t_s = intersect_scene(shadow_orig, light_dir)
            
            in_shadow = False
            # 如果射到了物体，并且交点距离小于光源距离，则有遮挡
            if hit_s != 0 and t_s < ti.math.length(LIGHT_POS - p):
                in_shadow = True
            
            ambient = ka * LIGHT_COLOR * obj_color
            diffuse = ti.math.vec3(0.0)
            specular = ti.math.vec3(0.0)
            
            if not in_shadow:
                diffuse = kd * LIGHT_COLOR * ti.max(0.0, n.dot(light_dir)) * obj_color
                if use_blinn == 1:
                    H = (light_dir + view_dir).normalized()
                    specular = ks * LIGHT_COLOR * (ti.max(0.0, n.dot(H)) ** shininess)
                else:
                    R = (2.0 * n.dot(light_dir) * n - light_dir).normalized()
                    specular = ks * LIGHT_COLOR * (ti.max(0.0, R.dot(view_dir)) ** shininess)
            
            color = ambient + diffuse + specular
        pixel_buffer[i, j] = ti.math.clamp(color, 0.0, 1.0)

def run():
    window = ti.ui.Window("Lab 3: Phong vs Blinn-Phong", WINDOW_RES)
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    ka, kd, ks, shininess = DEFAULT_KA, DEFAULT_KD, DEFAULT_KS, DEFAULT_SHININESS
    use_blinn = True

    while window.running:
        render(ka, kd, ks, shininess, 1 if use_blinn else 0)
        canvas.set_image(pixel_buffer)
        
        with gui.sub_window("Control Panel", 0.02, 0.02, 0.35, 0.3):
            ka = gui.slider_float("Ka", ka, 0.0, 1.0)
            kd = gui.slider_float("Kd", kd, 0.0, 1.0)
            ks = gui.slider_float("Ks", ks, 0.0, 1.0)
            shininess = gui.slider_float("N", shininess, 1.0, 128.0)
            
            if gui.button("Toggle Model"):
                use_blinn = not use_blinn
            
            gui.text(f"Current Model: {'Blinn-Phong' if use_blinn else 'Phong'}")
            gui.text("Shadows & Multi-Object: Active")
            
        window.show()

if __name__ == '__main__':
    run()