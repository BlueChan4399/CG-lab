import numpy as np


def de_casteljau(points, t):
    # De Casteljau 算法：通过递归线性插值计算 Bézier 曲线上参数为 t 的点。
    pts = points.copy()
    n = pts.shape[0]
    while n > 1:
        for i in range(n - 1):
            pts[i] = (1.0 - t) * pts[i] + t * pts[i + 1]
        n -= 1
    return float(pts[0, 0]), float(pts[0, 1])


def de_casteljau_levels(points, t):
    # 返回 De Casteljau 算法每一层的插值点，用于包络线动画可视化。
    pts = np.array(points, dtype=np.float32)
    n = pts.shape[0]
    levels = [pts.copy()]
    while n > 1:
        for i in range(n - 1):
            pts[i] = (1.0 - t) * pts[i] + t * pts[i + 1]
        n -= 1
        levels.append(pts[:n].copy())
    return levels


def sample_bezier(control_points, num_samples):
    # 在 Bézier 曲线上均匀采样 num_samples 个点。
    if len(control_points) < 2:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.array(control_points, dtype=np.float32)
    coords = np.zeros((num_samples, 2), dtype=np.float32)

    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0.0
        coords[i] = de_casteljau(pts, t)

    return coords


def catmull_rom_to_bezier(points, tension=1.0):
    # 将 Catmull-Rom 样条插值点转换为一系列三次 Bézier 控制点。
    # 每个分段保证 C1 连续。
    n = len(points)
    if n < 2:
        return []

    segments = []
    for i in range(n - 1):
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(n - 1, i + 2)]

        b0 = p1
        b1 = [
            p1[0] + (p2[0] - p0[0]) / (6.0 * tension),
            p1[1] + (p2[1] - p0[1]) / (6.0 * tension),
        ]
        b2 = [
            p2[0] - (p3[0] - p1[0]) / (6.0 * tension),
            p2[1] - (p3[1] - p1[1]) / (6.0 * tension),
        ]
        b3 = p2
        segments.append([b0, b1, b2, b3])

    return segments
