import numpy as np

# 窗口与显示
WINDOW_RES = (700, 700)
ASPECT_RATIO = WINDOW_RES[0] / WINDOW_RES[1]

# 相机与投影
EYE_POS = np.array([0.0, 0.0, 5.0], dtype=np.float32)
EYE_FOV = 45.0
Z_NEAR = 0.1
Z_FAR = 50.0

# 初始三角形顶点
VERTICES = np.array([
    [ 2.0,  0.0, -2.0, 1.0],
    [ 0.0,  2.0, -2.0, 1.0],[-2.0,  0.0, -2.0, 1.0]
], dtype=np.float32)

# 颜色
COLOR_GREEN = 0x00FF00
COLOR_BLUE = 0x0000FF
COLOR_RED = 0xFF0000