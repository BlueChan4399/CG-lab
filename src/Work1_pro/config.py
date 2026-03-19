import numpy as np

# 窗口与显示
WINDOW_RES = (700, 700)
ASPECT_RATIO = WINDOW_RES[0] / WINDOW_RES[1]

# 相机与投影
EYE_POS = np.array([0.0, 0.0, 5.0], dtype=np.float32)
EYE_FOV = 45.0
Z_NEAR = 0.1
Z_FAR = 50.0

# 正方体 8 个顶点，中心在原点，边长为 2（坐标范围 [-1, 1]）
CUBE_VERTICES = np.array([
    [-1.0, -1.0, -1.0, 1.0],
    [ 1.0, -1.0, -1.0, 1.0],
    [ 1.0,  1.0, -1.0, 1.0],
    [-1.0,  1.0, -1.0, 1.0],
    [-1.0, -1.0,  1.0, 1.0],
    [ 1.0, -1.0,  1.0, 1.0],
    [ 1.0,  1.0,  1.0, 1.0],
    [-1.0,  1.0,  1.0, 1.0],
], dtype=np.float32)

# 正方体 12 条边
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 前面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 后面
    (0, 4), (1, 5), (2, 6), (3, 7),  # 连接前后
]

# 颜色
COLOR_GREEN = 0x00FF00
COLOR_CYAN  = 0x00FFFF
COLOR_WHITE = 0xFFFFFF
