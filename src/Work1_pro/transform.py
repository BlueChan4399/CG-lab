import numpy as np
import math

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
