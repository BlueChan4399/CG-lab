import taichi as ti

from .config import (WINDOW_RES, ASPECT_RATIO, EYE_POS,
                     EYE_FOV, Z_NEAR, Z_FAR,
                     CUBE_VERTICES, CUBE_EDGES,
                     COLOR_GREEN, COLOR_CYAN, COLOR_WHITE)
from .transform import (get_model_matrix_y, get_model_matrix_x,
                         get_view_matrix, get_projection_matrix)

ti.init(arch=ti.cpu)

def run():
    print("正在启动 Work1_pro: 3D Cube Wireframe...")
    gui = ti.GUI("Experiment 1 Pro: 3D Cube Wireframe", res=WINDOW_RES)

    angle_y = 0.0
    angle_x = 0.0
    auto_rotate = True

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
            elif e.key == ti.GUI.ESCAPE:
                gui.running = False

        # 自动旋转
        if auto_rotate:
            angle_y += 0.5

        # MVP 矩阵
        model = get_model_matrix_y(angle_y) @ get_model_matrix_x(angle_x)
        view = get_view_matrix(EYE_POS)
        proj = get_projection_matrix(EYE_FOV, ASPECT_RATIO, Z_NEAR, Z_FAR)
        mvp = proj @ view @ model

        # 将所有顶点变换到屏幕坐标
        screen_coords = []
        for i in range(8):
            v = CUBE_VERTICES[i]
            v_clip = mvp @ v
            v_ndc = v_clip / v_clip[3]

            screen_x = 0.5 * v_ndc[0] + 0.5
            screen_y = 0.5 * v_ndc[1] + 0.5
            screen_coords.append([screen_x, screen_y])

        # 绘制 12 条边
        edge_colors = [COLOR_GREEN] * 4 + [COLOR_CYAN] * 4 + [COLOR_WHITE] * 4
        for idx, (a, b) in enumerate(CUBE_EDGES):
            gui.line(screen_coords[a], screen_coords[b],
                     radius=2, color=edge_colors[idx])

        gui.show()

if __name__ == '__main__':
    run()
