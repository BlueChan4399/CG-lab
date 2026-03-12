import taichi as ti

ti.init(arch=ti.gpu)

from .config import WINDOW_RES, PARTICLE_COLOR, PARTICLE_RADIUS
from .physics import init_particles, update_particles, pos

def run():
    print("正在编译 GPU 内核...")
    init_particles()
    
    gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES)
    print("编译完成！请在弹出的窗口中移动鼠标。")
    
    while gui.running:
        mouse_x, mouse_y = gui.get_cursor_pos()
        
        update_particles(mouse_x, mouse_y)
        
        gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
        gui.show()

if __name__ == "__main__":
    run()