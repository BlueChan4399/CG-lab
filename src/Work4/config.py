import taichi as ti

# ==== 窗口分辨率 ====
WINDOW_RES = (800, 600)

# ==== 摄像机参数 ====
CAMERA_POS = ti.math.vec3(0.0, 0.8, 5.5)
FOV = 60.0  # 垂直视场角（度）

# ==== 光源默认参数 (任务4：UI 滑动条会实时覆盖此值) ====
LIGHT_POS_DEFAULT = ti.math.vec3(1.5, 4.0, 2.0)
LIGHT_COLOR = ti.math.vec3(1.0, 1.0, 1.0)

# ==== 背景颜色 ====
BG_COLOR = ti.math.vec3(0.05, 0.06, 0.12)  # 深蓝夜空

# ==== 任务1：场景几何（隐式定义） ====
# 物体 ID 编号：
#   0 = 未命中
#   1 = 无限大平面 (Ground Plane，漫反射，黑白棋盘格)
#   2 = 红色漫反射球 (Red Diffuse Sphere)
#   3 = 银色镜面球   (Silver Mirror Sphere)
#   4 = 玻璃球       (Glass Sphere，选做：折射 + Snell's Law)

# (1) 无限大平面: y = -1.0，法线 (0,1,0)，黑白棋盘格
PLANE_Y = -1.0
PLANE_NORMAL = ti.math.vec3(0.0, 1.0, 0.0)
PLANE_COLOR_A = ti.math.vec3(0.95, 0.95, 0.95)  # 棋盘亮格
PLANE_COLOR_B = ti.math.vec3(0.10, 0.10, 0.10)  # 棋盘暗格

# (2) 红色漫反射球
RED_SPHERE_CENTER = ti.math.vec3(-1.5, 0.0, 0.0)
RED_SPHERE_RADIUS = 1.0
RED_SPHERE_COLOR = ti.math.vec3(0.85, 0.15, 0.15)

# (3) 银色纯镜面球
MIRROR_SPHERE_CENTER = ti.math.vec3(1.5, 0.0, 0.0)
MIRROR_SPHERE_RADIUS = 1.0
MIRROR_REFLECTIVITY = 0.8  # 镜面反射率（吞吐衰减系数）

# (4) 玻璃球 (选做：折射)
GLASS_SPHERE_CENTER = ti.math.vec3(0.0, -0.4, 2.2)
GLASS_SPHERE_RADIUS = 0.6
GLASS_IOR = 1.5                                  # 折射率（典型玻璃）
GLASS_TINT = ti.math.vec3(0.97, 0.99, 1.0)       # 透射衰减/微蓝染色

# ==== Phong 着色参数（用于漫反射物体） ====
KA = 0.15   # 环境光
KD = 0.75   # 漫反射
KS = 0.40   # 镜面高光（漫反射物体表面的微高光）
SHININESS = 32.0

# ==== 数值精度修正：Shadow Acne 偏移量 ====
EPSILON = 1e-4

# ==== 任务4：UI 滑动条范围 ====
LIGHT_RANGE = (-8.0, 8.0)       # 光源 X / Z 范围
LIGHT_Y_RANGE = (0.5, 10.0)     # 光源 Y 范围（保持在地面以上）
MAX_BOUNCES_RANGE = (1, 8)      # 加大上限以容纳玻璃 + 镜面的多次弹射
MAX_BOUNCES_DEFAULT = 4         # 默认 4：足以一次性穿透玻璃球（前/后两次折射）后再被反射

# ==== 选做：MSAA 抗锯齿样本数 ====
SAMPLES_RANGE = (1, 8)
SAMPLES_DEFAULT = 1             # 默认 1（关闭 AA），UI 可拉到 8 观察平滑边缘