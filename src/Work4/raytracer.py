import taichi as ti

@ti.func
def ray_sphere_intersect(orig, dir, center, radius):
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
        if t1 > 1e-4: t = t1; hit = True
        elif t2 > 1e-4: t = t2; hit = True
    return hit, t

@ti.func
def ray_cone_intersect(orig, dir, apex, base_y, radius):
    hit = False
    t = 1e20
    h = apex.y - base_y
    m = (radius / h) ** 2
    co = orig - apex
    
    a = dir.x**2 + dir.z**2 - m * dir.y**2
    b = 2.0 * (dir.x * co.x + dir.z * co.z - m * dir.y * co.y)
    c = co.x**2 + co.z**2 - m * co.y**2
    
    if ti.abs(a) > 1e-6:  # 核心修复：防止阴影射线的方向导致除以 0
        det = b*b - 4.0*a*c
        if det >= 0.0:
            sqrt_det = ti.math.sqrt(det)
            t1 = (-b - sqrt_det) / (2.0 * a)
            t2 = (-b + sqrt_det) / (2.0 * a)
            
            if t1 > 1e-4 and t1 < t:
                p1 = orig + t1 * dir
                if base_y <= p1.y <= apex.y:
                    t = t1
                    hit = True
            if t2 > 1e-4 and t2 < t:
                p2 = orig + t2 * dir
                if base_y <= p2.y <= apex.y:
                    t = t2
                    hit = True

    if ti.abs(dir.y) > 1e-6:
        t_b = (base_y - orig.y) / dir.y
        if t_b > 1e-4 and t_b < t:
            p_b = orig + t_b * dir
            if (p_b.x - apex.x)**2 + (p_b.z - apex.z)**2 <= radius**2:
                t = t_b
                hit = True
    return hit, t

@ti.func
def ray_plane_intersect(orig, dir, plane_y):
    hit = False
    t = 1e20
    if ti.abs(dir.y) > 1e-6:
        t_temp = (plane_y - orig.y) / dir.y
        if t_temp > 1e-4:
            t = t_temp
            hit = True
    return hit, t

@ti.func
def get_cone_normal(p, apex, base_y, radius):
    n = ti.math.vec3(0.0)
    if p.y <= base_y + 1e-3:
        n = ti.math.vec3(0.0, -1.0, 0.0)
    else:
        h = apex.y - base_y
        m = (radius / h) ** 2
        n = ti.math.vec3(p.x - apex.x, -m * (p.y - apex.y), p.z - apex.z)
        n = n.normalized()
    return n