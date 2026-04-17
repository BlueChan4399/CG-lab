import taichi as ti

# 窗口分辨率 (推荐保持该分辨率，Taichi加速后可丝滑运行)
WINDOW_RES = (800, 600)

# 摄像机与光源
CAMERA_POS = ti.math.vec3(0.0, 0.5, 5.0)  # 稍微调高一点相机，看地面更清楚
LIGHT_POS = ti.math.vec3(-4.0, 6.0, 2.0)   # 调高光源，让阴影投射在地面上
LIGHT_COLOR = ti.math.vec3(1.0, 1.0, 1.0)
BG_COLOR = ti.math.vec3(0.05, 0.05, 0.1)  # 背景改深一点，突出物体

# ==== 任务1: 构建几何体隐式定义 ====
# 红色球体 (Red Sphere)
SPHERE_CENTER = ti.math.vec3(-1.2, -0.2, 0.0)
SPHERE_RADIUS = 1.2
SPHERE_COLOR = ti.math.vec3(0.1, 0.6, 0.9)

# 紫色圆锥 (Purple Cone)
CONE_APEX = ti.math.vec3(1.2, 1.2, 0.0)  # 顶点
CONE_BASE_Y = -1.4                       # 底面高度
CONE_RADIUS = 1.2                        # 底面半径
CONE_COLOR = ti.math.vec3(0.9, 0.4, 0.1)

# ---- 增加地面配置 (为了体现阴影选做项) ----
PLANE_Y = -1.4
PLANE_COLOR = ti.math.vec3(0.4, 0.4, 0.4) # 灰色地面

# ==== 任务4: 材质参数默认值 ====
DEFAULT_KA = 0.2
DEFAULT_KD = 0.7
DEFAULT_KS = 0.5
DEFAULT_SHININESS = 32.0