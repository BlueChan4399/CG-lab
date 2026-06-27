import taichi as ti

# 窗口分辨率
WINDOW_RES = (1024, 768)

# task 1：布料网格规模
N = 20                          # 每边的质点数量（N×N）
QUAD_SIZE = 0.05                # 相邻质点的初始静止距离
CLOTH_OFFSET = ti.math.vec3(    # 布料左上角在世界坐标系中的位置
    -N * QUAD_SIZE * 0.5,       # X 居中
    0.45,                       # Y 高度
    -N * QUAD_SIZE * 0.5,       # Z 居中
)

# task 1：质点 / 弹簧物理参数
PARTICLE_MASS = 5.0e-2          # 每个质点的质量
SPRING_STIFFNESS = 6.0e2        # 弹簧劲度系数 k_s
DAMPING_DEFAULT = 0.2           # 阻尼系数 k_d
GRAVITY = ti.math.vec3(0.0, -9.8, 0.0)

# task 3：时间步与子步策略
DT = 1.6e-2                     # 单帧总仿真时间
SUBSTEPS = 30                   # 单帧内的子步数
IMPLICIT_ITERS = 30              # 隐式欧拉定点迭代次数

# task 2：防爆参数
MAX_VELOCITY = 50.0             # 速度上限

# task 3：结构 + 剪切 + 弯曲
SPRING_OFFSETS = [
    (1, 0), (0, 1),
    (1, 1), (1, -1),
    (2, 0), (0, 2),
]
NUM_SPRING_OFFSETS = len(SPRING_OFFSETS)

# 固定 4 个角点
PIN_POINTS = [
    (0,     0),
    (0,     N - 1),
    (N - 1, 0),
    (N - 1, N - 1),
]

# 渲染参数
CAMERA_POS_INIT = ti.math.vec3(0.0, 0.55, 1.2)
CAMERA_LOOKAT = ti.math.vec3(0.0, 0.10, 0.0)
CAMERA_UP = ti.math.vec3(0.0, 1.0, 0.0)
CAMERA_FOV = 55.0

BG_COLOR = (0.08, 0.10, 0.16)
PARTICLE_COLOR = (0.20, 0.55, 0.95)
PARTICLE_RADIUS = 0.010
POINT_LIGHT_POS = (0.6, 1.6, 1.2)
POINT_LIGHT_COLOR = (1.0, 1.0, 1.0)
AMBIENT_COLOR = (0.35, 0.35, 0.45)

# 积分方法枚举
METHOD_EXPLICIT = 0
METHOD_SEMI_IMPLICIT = 1
METHOD_IMPLICIT = 2
METHOD_NAMES = {
    METHOD_EXPLICIT:      "Explicit Euler (Unstable)",
    METHOD_SEMI_IMPLICIT: "Semi-Implicit Euler (Stable)",
    METHOD_IMPLICIT:      "Implicit Euler (Damped)",
}

# UI 滑动条范围
DAMPING_RANGE = (0.0, 5.0)
STIFFNESS_RANGE = (2.0e2, 2.5e3)
