import numpy as np

# 窗口与光栅化分辨率
WINDOW_RES = (1024, 1024)

# 最大控制点数量
MAX_CONTROL_POINTS = 100

# Bézier 曲线采样点数
NUM_SAMPLES = 1000

# 反走样与动画
AA_SCALE = 2
NUM_SAMPLES_SPLINE = 500
ENVELOPE_SPEED = 0.005
DRAG_RADIUS = 15.0

# 颜色配置
COLOR_POLY = 0x808080     # 多边形（灰）
COLOR_POINT = 0x0000FF    # 控制点（蓝）
COLOR_ENVELOPE = 0x00FF00 # 包络线（绿）
COLOR_CURVE_POINT = 0xFF0000  # 包络动画尖端点（红）
