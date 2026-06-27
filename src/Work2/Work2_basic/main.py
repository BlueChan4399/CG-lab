import taichi as ti

from .config import (WINDOW_RES, ASPECT_RATIO, EYE_POS, 
                     EYE_FOV, Z_NEAR, Z_FAR, VERTICES,
                     COLOR_GREEN, COLOR_BLUE, COLOR_RED)
from .transform import get_model_matrix, get_view_matrix, get_projection_matrix

ti.init(arch=ti.cpu)

def run():
    print("正在启动 Work1: 3D Transformation...")
    gui = ti.GUI("Experiment 1: 3D Transformation", res=WINDOW_RES)
    
    angle = 0.0

    while gui.running:
        # 按键
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in ['a', 'A']:
                angle += 5.0
            elif e.key in ['d', 'D']:
                angle -= 5.0
            elif e.key == ti.GUI.ESCAPE:
                gui.running = False
                
        # 获取三个矩阵
        model = get_model_matrix(angle)
        view = get_view_matrix(EYE_POS)
        proj = get_projection_matrix(EYE_FOV, ASPECT_RATIO, Z_NEAR, Z_FAR)
        
        # MVP 矩阵相乘
        mvp = proj @ view @ model
        
        # 计算顶点屏幕坐标
        screen_coords =[]
        for i in range(3):
            v = VERTICES[i]
            v_clip = mvp @ v                     
            v_ndc = v_clip / v_clip[3]           
            
            screen_x = 0.5 * v_ndc[0] + 0.5
            screen_y = 0.5 * v_ndc[1] + 0.5
            screen_coords.append([screen_x, screen_y])
            
        # 绘制三角形
        gui.line(screen_coords[0], screen_coords[1], radius=2, color=COLOR_GREEN)
        gui.line(screen_coords[1], screen_coords[2], radius=2, color=COLOR_BLUE)
        gui.line(screen_coords[2], screen_coords[0], radius=2, color=COLOR_RED)

        gui.show()

if __name__ == '__main__':
    run()