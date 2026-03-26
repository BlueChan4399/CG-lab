import numpy as np
import math


def euler_to_quaternion(angle_x, angle_y, angle_z=0.0):
    rx = math.radians(angle_x) / 2.0
    ry = math.radians(angle_y) / 2.0
    rz = math.radians(angle_z) / 2.0

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return np.array([w, x, y, z], dtype=np.float32)


def slerp(q0, q1, t):
    dot = np.dot(q0, q1)

    # 确保走最短路径
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(dot, 1.0)

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def quaternion_to_model_matrix(q):
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y), 0.0],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x), 0.0],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y), 0.0],
        [              0.0,               0.0,               0.0,  1.0]
    ], dtype=np.float32)



def get_model_matrix_y(angle):
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    model = np.array([
        [ cos_a, 0.0, sin_a, 0.0],
        [   0.0, 1.0,   0.0, 0.0],
        [-sin_a, 0.0, cos_a, 0.0],
        [   0.0, 0.0,   0.0, 1.0]
    ], dtype=np.float32)
    return model

def get_model_matrix_x(angle):

    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    model = np.array([
        [1.0,   0.0,    0.0, 0.0],
        [0.0, cos_a, -sin_a, 0.0],
        [0.0, sin_a,  cos_a, 0.0],
        [0.0,   0.0,    0.0, 1.0]
    ], dtype=np.float32)
    return model

def get_view_matrix(eye_pos):

    view = np.array([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return view

def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):

    n = -zNear
    f = -zFar

    M_persp2ortho = np.array([
        [n,   0.0, 0.0,     0.0],
        [0.0, n,   0.0,     0.0],
        [0.0, 0.0, n + f,  -n * f],
        [0.0, 0.0, 1.0,     0.0]
    ], dtype=np.float32)

    rad = math.radians(eye_fov)
    t = math.tan(rad / 2.0) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    M_ortho_scale = np.array([
        [2.0 / (r - l), 0.0,           0.0,           0.0],
        [0.0,           2.0 / (t - b), 0.0,           0.0],
        [0.0,           0.0,           2.0 / (n - f), 0.0],
        [0.0,           0.0,           0.0,           1.0]
    ], dtype=np.float32)

    M_ortho_trans = np.array([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    M_ortho = M_ortho_scale @ M_ortho_trans

    projection = M_ortho @ M_persp2ortho
    return projection
