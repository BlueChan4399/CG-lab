import numpy as np

# 窗口与显示
WINDOW_RES = (700, 700)
ASPECT_RATIO = WINDOW_RES[0] / WINDOW_RES[1]

# 相机与投影
EYE_POS = np.array([0.0, 0.0, 5.0], dtype=np.float32)
EYE_FOV = 45.0
Z_NEAR = 0.1
Z_FAR = 50.0

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

CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  
    (4, 5), (5, 6), (6, 7), (7, 4),  
    (0, 4), (1, 5), (2, 6), (3, 7),  
]

# 颜色
COLOR_GREEN = 0x00FF00
COLOR_CYAN  = 0x00FFFF
COLOR_WHITE = 0xFFFFFF
COLOR_ORANGE = 0xFF8800

# 旋转插值
POSE_R0 = (0.0, 0.0)        # 姿态R0：正面朝前
POSE_R1 = (45.0, 90.0)      # 姿态R1：绕X轴45°+绕Y轴90°
SLERP_SPEED = 0.09          # 插值速度
