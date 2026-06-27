import math

import taichi as ti
import numpy as np

from .config import (WINDOW_RES, MAX_CONTROL_POINTS, NUM_SAMPLES, NUM_SAMPLES_SPLINE,
                     AA_SCALE, ENVELOPE_SPEED, DRAG_RADIUS,
                     COLOR_POLY, COLOR_POINT, COLOR_ENVELOPE, COLOR_CURVE_POINT)
from .bezier import sample_bezier, de_casteljau_levels, catmull_rom_to_bezier

ti.init(arch=ti.gpu)

W, H = WINDOW_RES

# 初始化 CPU 画布与 GPU 像素缓冲
canvas = np.zeros(WINDOW_RES, dtype=np.float32)
canvas_aa = np.zeros((W * AA_SCALE, H * AA_SCALE), dtype=np.float32)
pixel_field = ti.field(ti.f32, shape=WINDOW_RES)

# 对象池
gui_points = ti.Vector.field(2, ti.f32, shape=MAX_CONTROL_POINTS)
gui_lines_begin = ti.Vector.field(2, ti.f32, shape=MAX_CONTROL_POINTS)
gui_lines_end = ti.Vector.field(2, ti.f32, shape=MAX_CONTROL_POINTS)


@ti.kernel
def draw_pixels_kernel(pixels: ti.types.ndarray(ndim=2), x: ti.f32, y: ti.f32):
    # 将单个像素 (x, y) 绘制到缓冲中。
    px = int(ti.round(x - 0.5))
    py = int(ti.round(y - 0.5))
    if 0 <= px < pixels.shape[0] and 0 <= py < pixels.shape[1]:
        pixels[px, py] = 1.0


@ti.kernel
def draw_batch_kernel(pixels: ti.types.ndarray(ndim=2), coords: ti.types.ndarray(ndim=2)):
    for i in range(coords.shape[0]):
        px = int(ti.round(coords[i, 0] - 0.5))
        py = int(ti.round(coords[i, 1] - 0.5))
        if 0 <= px < pixels.shape[0] and 0 <= py < pixels.shape[1]:
            pixels[px, py] = 1.0


@ti.kernel
def clear_canvas(pixels: ti.types.ndarray(ndim=2)):
    # 清空像素缓冲
    for i, j in pixels:
        pixels[i, j] = 0.0


def run():
    control_points = []
    current_mode = 'bezier'
    anti_aliasing = False
    dragging = False
    drag_idx = -1
    t_envelope = 0.0

    gui = ti.GUI("Lab2: Bézier Curve Rasterization", res=WINDOW_RES)

    while gui.running:
        mx, my = gui.get_cursor_pos()
        px = mx * W
        py = my * H

        # ---- 事件处理 ----
        for e in gui.get_events():
            if e.type == ti.GUI.PRESS:
                if e.key == ti.GUI.LMB:
                    nearest_idx = -1
                    nearest_dist = float('inf')
                    for i, cp in enumerate(control_points):
                        d = math.hypot(cp[0] - px, cp[1] - py)
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_idx = i
                    if nearest_idx >= 0 and nearest_dist < DRAG_RADIUS:
                        dragging = True
                        drag_idx = nearest_idx
                    else:
                        if len(control_points) < MAX_CONTROL_POINTS:
                            control_points.append([px, py])
                elif e.key == ti.GUI.RMB:
                    current_mode = 'spline' if current_mode == 'bezier' else 'bezier'
                elif e.key == ' ':
                    anti_aliasing = not anti_aliasing
                elif e.key in ['c', 'C']:
                    control_points.clear()
                elif e.key == ti.GUI.ESCAPE:
                    gui.running = False
            elif e.type == ti.GUI.RELEASE:
                if e.key == ti.GUI.LMB:
                    dragging = False
                    drag_idx = -1

        if dragging and drag_idx >= 0:
            control_points[drag_idx] = [px, py]

        # 光栅化
        scale = AA_SCALE if anti_aliasing else 1
        target_canvas = canvas_aa if anti_aliasing else canvas
        clear_canvas(target_canvas)

        n = len(control_points)
        if current_mode == 'bezier':
            if n >= 2:
                coords = sample_bezier(control_points, NUM_SAMPLES)
                coords = coords * scale
                draw_batch_kernel(target_canvas, coords.astype(np.float32))
            elif n == 1:
                draw_pixels_kernel(target_canvas, control_points[0][0] * scale,
                                   control_points[0][1] * scale)
        else:  # spline mode
            if n >= 2:
                segments = catmull_rom_to_bezier(control_points)
                for seg in segments:
                    coords = sample_bezier(seg, NUM_SAMPLES_SPLINE)
                    coords = coords * scale
                    draw_batch_kernel(target_canvas, coords.astype(np.float32))

        if anti_aliasing:
            downsampled = canvas_aa.reshape(H, AA_SCALE, W, AA_SCALE).mean(axis=(1, 3))
            pixel_field.from_numpy(downsampled)
        else:
            pixel_field.from_numpy(canvas)
        gui.set_image(pixel_field)

        # 对象池更新：控制点与控制多边形
        np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        np_lines_begin = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        np_lines_end = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)

        for i in range(n):
            np_points[i] = [control_points[i][0] / W, control_points[i][1] / H]
            if i < n - 1:
                np_lines_begin[i] = np_points[i]
                np_lines_end[i] = [
                    control_points[i + 1][0] / W,
                    control_points[i + 1][1] / H
                ]

        gui_points.from_numpy(np_points)
        gui_lines_begin.from_numpy(np_lines_begin)
        gui_lines_end.from_numpy(np_lines_end)

        if n > 0:
            gui.circles(gui_points.to_numpy()[:n], radius=4, color=COLOR_POINT)
        if n > 1:
            gui.lines(gui_lines_begin.to_numpy()[:n - 1],
                      gui_lines_end.to_numpy()[:n - 1], radius=1, color=COLOR_POLY)

        # 包络线动画
        if n >= 2:
            if current_mode == 'bezier':
                envelope_source = control_points
            else:
                segments = catmull_rom_to_bezier(control_points)
                envelope_source = segments[0] if segments else None

            if envelope_source:
                levels = de_casteljau_levels(envelope_source, t_envelope)
                for level_pts in levels:
                    m = len(level_pts)
                    if m >= 2:
                        begins = np.array(level_pts[:-1], dtype=np.float32)
                        ends = np.array(level_pts[1:], dtype=np.float32)
                        begins[:, 0] /= W
                        begins[:, 1] /= H
                        ends[:, 0] /= W
                        ends[:, 1] /= H
                        gui.lines(begins, ends, radius=1, color=COLOR_ENVELOPE)
                # 尖端点
                tip = np.array([[levels[-1][0, 0] / W, levels[-1][0, 1] / H]], dtype=np.float32)
                gui.circles(tip, radius=3, color=COLOR_CURVE_POINT)

            t_envelope = (t_envelope + ENVELOPE_SPEED) % 1.0

        # UI
        mode_text = "Bezier" if current_mode == 'bezier' else "Spline"
        aa_text = "AA:ON" if anti_aliasing else "AA:OFF"
        gui.text(
            f"Mode={mode_text} | {aa_text} | LMB:add/drag | RMB:mode | Space:AA | C:clear",
            pos=(0.01, 0.97),
            font_size=13,
            color=0xFFFFFF,
        )

        gui.show()


if __name__ == '__main__':
    run()
