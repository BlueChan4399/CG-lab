import taichi as ti

from .config import (WINDOW_RES, ASPECT_RATIO, EYE_POS,
                     EYE_FOV, Z_NEAR, Z_FAR,
                     CUBE_VERTICES, CUBE_EDGES,
                     COLOR_GREEN, COLOR_CYAN, COLOR_WHITE, COLOR_ORANGE,
                     POSE_R0, POSE_R1, SLERP_SPEED)
from .transform import (get_model_matrix_y, get_model_matrix_x,
                         get_view_matrix, get_projection_matrix,
                         euler_to_quaternion, slerp, quaternion_to_model_matrix)

ti.init(arch=ti.cpu)


def project_cube(mvp):
    screen_coords = []
    for i in range(8):
        v = CUBE_VERTICES[i]
        v_clip = mvp @ v
        v_ndc = v_clip / v_clip[3]

        screen_x = 0.5 * v_ndc[0] + 0.5
        screen_y = 0.5 * v_ndc[1] + 0.5
        screen_coords.append([screen_x, screen_y])
    return screen_coords


def draw_cube(gui, screen_coords, edge_colors):
    """绘制立方体 12 条边"""
    for idx, (a, b) in enumerate(CUBE_EDGES):
        gui.line(screen_coords[a], screen_coords[b],
                 radius=2, color=edge_colors[idx])


def run():
    print("W/S/A/D: 手动旋转 | Space: 切换自动旋转")
    print("I: 切换 SLERP 插值模式（在 R0 和 R1 两个姿态间平滑过渡）")
    gui = ti.GUI("Experiment 1 Pro: 3D Cube Wireframe + Rotation Interpolation",
                 res=WINDOW_RES)

    angle_y = 0.0
    angle_x = 0.0
    auto_rotate = True

    # SLERP 插值相关
    interp_mode = False         # 是否处于插值模式
    t = 0.0                     # 插值参数[0,1]
    t_direction = 1.0           # 1.0正向,-1.0反向

    # 预计算 R0 和 R1 的四元数
    q0 = euler_to_quaternion(POSE_R0[0], POSE_R0[1])
    q1 = euler_to_quaternion(POSE_R1[0], POSE_R1[1])

    while gui.running:
        # 按键处理
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in ['a', 'A']:
                angle_y += 5.0
            elif e.key in ['d', 'D']:
                angle_y -= 5.0
            elif e.key in ['w', 'W']:
                angle_x += 5.0
            elif e.key in ['s', 'S']:
                angle_x -= 5.0
            elif e.key == ' ':
                auto_rotate = not auto_rotate
            elif e.key in ['i', 'I']:
                interp_mode = not interp_mode
                if interp_mode:
                    t = 0.0
                    t_direction = 1.0
                    auto_rotate = False
                    print("SLERP 插值模式：开启")
                else:
                    print("SLERP 插值模式：关闭")
            elif e.key == ti.GUI.ESCAPE:
                gui.running = False

        view = get_view_matrix(EYE_POS)
        proj = get_projection_matrix(EYE_FOV, ASPECT_RATIO, Z_NEAR, Z_FAR)

        if interp_mode:
            # ===== SLERP 插值模式 =====
            # 更新 t，来回循环
            t += SLERP_SPEED * t_direction
            if t >= 1.0:
                t = 1.0
                t_direction = -1.0
            elif t <= 0.0:
                t = 0.0
                t_direction = 1.0

            # SLERP 插值得到当前旋转四元数，转为模型矩阵
            q_current = slerp(q0, q1, t)
            model = quaternion_to_model_matrix(q_current)
            mvp = proj @ view @ model

            screen_coords = project_cube(mvp)
            interp_colors = [COLOR_ORANGE] * 12
            draw_cube(gui, screen_coords, interp_colors)

            model_r0 = quaternion_to_model_matrix(q0)
            mvp_r0 = proj @ view @ model_r0
            coords_r0 = project_cube(mvp_r0)
            r0_colors = [0x004400] * 12  # 暗绿
            draw_cube(gui, coords_r0, r0_colors)

            model_r1 = quaternion_to_model_matrix(q1)
            mvp_r1 = proj @ view @ model_r1
            coords_r1 = project_cube(mvp_r1)
            r1_colors = [0x004444] * 12 
            draw_cube(gui, coords_r1, r1_colors)

            # 显示插值进度
            gui.text(f"SLERP t = {t:.2f}", pos=(0.02, 0.96), color=COLOR_ORANGE)
            gui.text("R0", pos=(0.02, 0.92), color=0x00AA00)
            gui.text("R1", pos=(0.08, 0.92), color=0x00AAAA)

        else:
            # 普通手动/自动旋转模式 
            if auto_rotate:
                angle_y += 0.5

            model = get_model_matrix_y(angle_y) @ get_model_matrix_x(angle_x)
            mvp = proj @ view @ model

            screen_coords = project_cube(mvp)
            edge_colors = [COLOR_GREEN] * 4 + [COLOR_CYAN] * 4 + [COLOR_WHITE] * 4
            draw_cube(gui, screen_coords, edge_colors)

        gui.show()

if __name__ == '__main__':
    run()
