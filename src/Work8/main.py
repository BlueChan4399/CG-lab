import argparse
import os
import sys
import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(SCRIPT_DIR, ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import smplx
from smplx.lbs import (
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    vertices2joints,
)





class _ChumpyArrayShim:
    """Minimal pickle shim for old SMPL files that store arrays as chumpy.Ch."""

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _array(self):
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x
        raise AttributeError("Cannot recover array data from chumpy pickle object")

    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)

    @property
    def shape(self):
        return np.asarray(self).shape

    def __len__(self):
        return len(np.asarray(self))

    def __getitem__(self, item):
        return np.asarray(self)[item]


def install_chumpy_pickle_shim():
    if "chumpy.ch" in sys.modules:
        return

    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")

    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module

    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def set_axes_equal(ax, vertices):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = np.max(maxs - mins) * 0.5 + 1e-8
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def smpl_to_plot_coords(points):
    return points[:, [0, 2, 1]]


def scalar_to_face_colors(vertex_scalar, faces, cmap_name="viridis"):
    scalar = vertex_scalar.astype(np.float64)
    scalar = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-8)
    face_scalar = scalar[faces].mean(axis=1)
    return plt.get_cmap(cmap_name)(face_scalar)


def joint_weight_face_colors(lbs_weights, faces):
    face_weights = lbs_weights[faces].mean(axis=1)
    dominant_joint = np.argmax(face_weights, axis=1)
    dominant_weight = np.max(face_weights, axis=1)

    palette = plt.get_cmap("hsv")(
        np.linspace(0.0, 1.0, lbs_weights.shape[1], endpoint=False)
    )
    face_colors = palette[dominant_joint]
    strength = 0.35 + 0.65 * dominant_weight
    face_colors[:, :3] = face_colors[:, :3] * strength[:, None]
    face_colors[:, :3] += (1.0 - strength[:, None]) * 0.88
    face_colors[:, 3] = 1.0
    return face_colors


def shade_face_colors(vertices, faces, face_colors):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

    light_dir = np.array([-0.25, -0.55, 0.80], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    intensity = 0.35 + 0.65 * np.clip(normals @ light_dir, 0.0, 1.0)

    shaded = face_colors.copy()
    shaded[:, :3] *= intensity[:, None]
    return shaded


def draw_mesh(
    ax,
    vertices,
    faces,
    joints=None,
    vertex_scalar=None,
    face_colors=None,
    title="",
    elev=12,
    azim=108,
    axis_bounds=None,
):
    plot_vertices = smpl_to_plot_coords(vertices)
    plot_joints = None if joints is None else smpl_to_plot_coords(joints)

    if face_colors is None:
        if vertex_scalar is None:
            face_colors = np.tile(np.array([[0.82, 0.67, 0.52, 1.0]]), (faces.shape[0], 1))
        else:
            face_colors = scalar_to_face_colors(vertex_scalar, faces)
    face_colors = shade_face_colors(plot_vertices, faces, face_colors.copy())

    mesh = Poly3DCollection(
        plot_vertices[faces],
        facecolors=face_colors,
        edgecolors=(0.0, 0.0, 0.0, 0.05),
        linewidths=0.03,
    )
    ax.add_collection3d(mesh)

    if plot_joints is not None:
        ax.scatter(
            plot_joints[:, 0],
            plot_joints[:, 1],
            plot_joints[:, 2],
            c="white",
            s=12,
            depthshade=False,
            edgecolors="black",
            linewidths=0.3,
        )

    if axis_bounds is None:
        set_axes_equal(ax, plot_vertices)
    else:
        center, radius = axis_bounds
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_proj_type("persp", focal_length=0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def save_single_figure(path, vertices, faces, joints=None, vertex_scalar=None, title=""):
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(ax, vertices, faces, joints=joints, vertex_scalar=vertex_scalar, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_comparison_grid(path, data, faces):
    fig = plt.figure(figsize=(14, 10))

    panels = [
        ("(a) Template + LBS Weights", data["v_template"], data["J_template"], data["weight_scalar"]),
        ("(b) Shape Blend + Joint Regression", data["v_shaped"], data["J_shaped"], None),
        ("(c) Pose Blend Shapes", data["v_posed"], data["J_shaped"], data["pose_offset_norm"]),
        ("(d) Final LBS Result", data["verts"], data["J_transformed"], None),
    ]

    for index, (title, vertices, joints, scalar) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        draw_mesh(ax, vertices, faces, joints=joints, vertex_scalar=scalar, title=title)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_all_joint_weights_figure(path, vertices, faces, joints, lbs_weights):
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(
        ax,
        vertices,
        faces,
        joints=joints,
        face_colors=joint_weight_face_colors(lbs_weights, faces),
        title="All Joint LBS Weights",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_demo_shape(device, dtype, num_betas):
    betas = torch.zeros((1, num_betas), dtype=dtype, device=device)
    if num_betas >= 1:
        betas[0, 0] = 2.0
    if num_betas >= 2:
        betas[0, 1] = -1.2
    if num_betas >= 3:
        betas[0, 2] = 0.8
    return betas


def build_demo_pose(device, dtype):
    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)

    joint_names = {
        "left_hip": 1,
        "right_hip": 2,
        "left_knee": 4,
        "right_knee": 5,
        "left_shoulder": 16,
        "right_shoulder": 17,
        "left_elbow": 18,
        "right_elbow": 19,
    }

    def set_pose(name, axis_angle):
        start = (joint_names[name] - 1) * 3
        body_pose[0, start:start + 3] = torch.tensor(axis_angle, dtype=dtype, device=device)

    set_pose("left_shoulder", [0.0, 0.0, 0.45])
    set_pose("right_shoulder", [0.0, 0.0, -0.45])
    set_pose("left_elbow", [0.0, -0.35, 0.0])
    set_pose("right_elbow", [0.0, 0.35, 0.0])
    set_pose("left_hip", [0.25, 0.0, 0.08])
    set_pose("right_hip", [-0.18, 0.0, -0.08])
    set_pose("left_knee", [0.35, 0.0, 0.0])
    set_pose("right_knee", [0.20, 0.0, 0.0])
    return global_orient, body_pose


AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def build_single_joint_pose(device, dtype, joint_id, axis, angle_rad):
    """Zero pose except for one joint rotated by `angle_rad` around `axis`.

    joint_id follows the full 24-joint SMPL order: 0 is the global root and
    1..23 map into body_pose at offset (joint_id - 1) * 3.
    """
    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)

    unit = AXIS_VECTORS[axis]
    axis_angle = torch.tensor([component * angle_rad for component in unit], dtype=dtype, device=device)

    if joint_id == 0:
        global_orient[0] = axis_angle
    else:
        start = (joint_id - 1) * 3
        body_pose[0, start:start + 3] = axis_angle
    return global_orient, body_pose


def prepare_posedirs(posedirs, expected_pose_dim):
    if posedirs.dim() != 2:
        posedirs = posedirs.reshape(posedirs.shape[0], -1)

    if posedirs.shape[0] == expected_pose_dim:
        return posedirs
    if posedirs.shape[1] == expected_pose_dim:
        return posedirs.T

    raise RuntimeError(
        f"posedirs shape {tuple(posedirs.shape)} does not match pose dimension "
        f"{expected_pose_dim}"
    )


def compute_manual_lbs(model, betas, global_orient, body_pose):
    device = betas.device
    dtype = betas.dtype

    v_template = model.v_template
    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)

    shapedirs = model.shapedirs[:, :, : betas.shape[1]]
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    J = vertices2joints(model.J_regressor, v_shaped)

    full_pose = torch.cat([global_orient, body_pose], dim=1)
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)

    identity = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:, :, :] - identity).view(1, -1)
    posedirs = prepare_posedirs(model.posedirs, expected_pose_dim=pose_feature.shape[1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets

    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)

    num_joints = J.shape[1]
    W = model.lbs_weights.unsqueeze(0).expand(1, -1, -1)
    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)

    ones = torch.ones((1, v_posed.shape[1], 1), dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, ones], dim=2)
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]

    J_template = vertices2joints(model.J_regressor, v_template)

    return {
        "v_template": v_template,
        "v_shaped": v_shaped,
        "J": J,
        "J_template": J_template,
        "pose_offsets": pose_offsets,
        "v_posed": v_posed,
        "J_transformed": J_transformed,
        "verts": verts,
    }


def compare_with_official_forward(model, betas, global_orient, body_pose, manual_verts):
    with torch.no_grad():
        output = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            return_verts=True,
        )
    diff = torch.abs(manual_verts - output.vertices)
    return diff.mean().item(), diff.max().item()


def assemble_gif(frame_paths, gif_path, fps=12, ping_pong=True):
    """Stitch saved frames into a looping GIF using Pillow.

    Frames must share identical pixel dimensions, so the animation renderer
    saves them with a fixed figure size and without bbox cropping.
    Returns True on success, False if Pillow is unavailable.
    """
    sequence = list(frame_paths)
    if ping_pong and len(frame_paths) > 2:
        sequence = list(frame_paths) + list(frame_paths[-2:0:-1])

    try:
        from PIL import Image
    except ImportError as exc:
        print(f"GIF assembly skipped ({exc}); individual frames are still saved.")
        return False

    frames = [Image.open(path).convert("RGB") for path in sequence]
    duration_ms = int(round(1000.0 / float(fps)))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return True


def _content_bbox(image, bg_threshold=250):
    """Bounding box (left, top, right, bottom) of the non-white content."""
    arr = np.asarray(image.convert("RGB"))
    mask = np.any(arr < bg_threshold, axis=2)
    if not mask.any():
        return (0, 0, image.width, image.height)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def crop_frames_to_common_bbox(frame_paths, pad=10):
    """Crop every frame to one shared rectangle (the union of their content),
    trimming the empty margins mpl 3D leaves while keeping all frames the same
    size so they can still be stacked into a GIF. No-op without Pillow."""
    try:
        from PIL import Image
    except ImportError:
        return

    images = [Image.open(path).convert("RGB") for path in frame_paths]
    boxes = [_content_bbox(image) for image in images]
    width, height = images[0].size
    left = max(min(box[0] for box in boxes) - pad, 0)
    top = max(min(box[1] for box in boxes) - pad, 0)
    right = min(max(box[2] for box in boxes) + pad, width)
    bottom = min(max(box[3] for box in boxes) + pad, height)
    crop_box = (left, top, right, bottom)

    for image, path in zip(images, frame_paths):
        image.crop(crop_box).save(path)


def save_pose_animation(
    out_dir,
    model,
    betas,
    faces,
    weight_scalar,
    joint_id,
    axis,
    max_angle_deg,
    num_frames,
    device,
    dtype,
    fps=12,
):
    """Optional content: fix the shape, swing one joint 0 -> max_angle, and
    render each frame colored by that joint's LBS weight so the weight region
    can be seen sliding smoothly with the bone."""
    frames_dir = os.path.join(out_dir, "animation_frames")
    os.makedirs(frames_dir, exist_ok=True)

    angles = np.linspace(0.0, np.radians(max_angle_deg), num_frames)

    # First pass: pose every frame and accumulate a shared bounding box so the
    # body stays centered instead of jittering frame to frame.
    frame_vertices = []
    frame_joints = []
    for angle in angles:
        global_orient, body_pose = build_single_joint_pose(device, dtype, joint_id, axis, float(angle))
        with torch.no_grad():
            data = compute_manual_lbs(model, betas, global_orient, body_pose)
        frame_vertices.append(to_numpy(data["verts"][0]))
        frame_joints.append(to_numpy(data["J_transformed"][0]))

    all_plot = smpl_to_plot_coords(np.concatenate(frame_vertices, axis=0))
    mins = all_plot.min(axis=0)
    maxs = all_plot.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = np.max(maxs - mins) * 0.5 + 1e-8
    axis_bounds = (center, radius)

    # Second pass: render with fixed axes / fixed figure size (no tight bbox so
    # every frame is the same resolution and can be stacked into a GIF).
    frame_paths = []
    for index, (vertices, joints, angle) in enumerate(zip(frame_vertices, frame_joints, angles)):
        fig = plt.figure(figsize=(5, 6))
        ax = fig.add_subplot(111, projection="3d")
        draw_mesh(
            ax,
            vertices,
            faces,
            joints=joints,
            vertex_scalar=weight_scalar,
            title=f"Joint {joint_id} rot {axis} = {np.degrees(angle):6.1f} deg",
            axis_bounds=axis_bounds,
        )
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.94)
        frame_path = os.path.join(frames_dir, f"frame_{index:03d}.png")
        fig.savefig(frame_path, dpi=140)
        plt.close(fig)
        frame_paths.append(frame_path)

    crop_frames_to_common_bbox(frame_paths)

    gif_path = os.path.join(out_dir, "pose_animation.gif")
    gif_ok = assemble_gif(frame_paths, gif_path, fps=fps)
    return {
        "frames_dir": frames_dir,
        "num_frames": num_frames,
        "frame_paths": frame_paths,
        "gif_path": gif_path if gif_ok else None,
    }


def main(args):
    model_path = resolve_path(args.model_path)
    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SMPL model file not found: {model_path}")

    device = torch.device("cpu")
    dtype = torch.float32

    install_chumpy_pickle_shim()
    model = smplx.create(
        model_path=model_path,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        num_betas=args.num_betas,
    ).to(device)

    faces = np.asarray(model.faces, dtype=np.int32)
    num_vertices = int(model.v_template.shape[0])
    num_faces = int(faces.shape[0])
    num_joints = int(model.lbs_weights.shape[1])

    betas = build_demo_shape(device, dtype, args.num_betas)
    global_orient, body_pose = build_demo_pose(device, dtype)
    data = compute_manual_lbs(model, betas, global_orient, body_pose)

    mean_err, max_err = compare_with_official_forward(
        model, betas, global_orient, body_pose, data["verts"]
    )

    joint_id = int(args.joint_id)
    if joint_id < 0 or joint_id >= num_joints:
        raise ValueError(f"joint-id must be in [0, {num_joints - 1}], got {joint_id}")

    weight_scalar = to_numpy(model.lbs_weights[:, joint_id])
    pose_offset_norm = np.linalg.norm(to_numpy(data["pose_offsets"][0]), axis=1)

    v_template = to_numpy(data["v_template"][0])
    v_shaped = to_numpy(data["v_shaped"][0])
    J_template = to_numpy(data["J_template"][0])
    J = to_numpy(data["J"][0])
    v_posed = to_numpy(data["v_posed"][0])
    J_transformed = to_numpy(data["J_transformed"][0])
    verts = to_numpy(data["verts"][0])

    save_single_figure(
        os.path.join(out_dir, "stage_a_template_weights.png"),
        v_template,
        faces,
        joints=J_template,
        vertex_scalar=weight_scalar,
        title=f"(a) Template Mesh + Weight of Joint {joint_id}",
    )
    save_single_figure(
        os.path.join(out_dir, "stage_b_shaped_joints.png"),
        v_shaped,
        faces,
        joints=J,
        title="(b) Shape Blend + Joint Regression",
    )
    save_single_figure(
        os.path.join(out_dir, "stage_c_pose_offsets.png"),
        v_posed,
        faces,
        joints=J,
        vertex_scalar=pose_offset_norm,
        title="(c) Pose Blend Shapes (colored by |pose_offsets|)",
    )
    save_single_figure(
        os.path.join(out_dir, "stage_d_lbs_result.png"),
        verts,
        faces,
        joints=J_transformed,
        title="(d) Final LBS Result",
    )
    save_comparison_grid(
        os.path.join(out_dir, "comparison_grid.png"),
        {
            "v_template": v_template,
            "J_template": J_template,
            "v_shaped": v_shaped,
            "J_shaped": J,
            "v_posed": v_posed,
            "J_transformed": J_transformed,
            "verts": verts,
            "weight_scalar": weight_scalar,
            "pose_offset_norm": pose_offset_norm,
        },
        faces,
    )
    save_all_joint_weights_figure(
        os.path.join(out_dir, "all_joint_weights.png"),
        v_template,
        faces,
        J_template,
        to_numpy(model.lbs_weights),
    )

    anim_info = None
    if not args.skip_animation:
        anim_joint_id = int(args.anim_joint_id)
        if anim_joint_id < 0 or anim_joint_id >= num_joints:
            raise ValueError(
                f"anim-joint-id must be in [0, {num_joints - 1}], got {anim_joint_id}"
            )
        # Color the animation by the rotating joint's own weights so the high
        # weight region is exactly the surface that gets dragged along.
        anim_weight_scalar = to_numpy(model.lbs_weights[:, anim_joint_id])
        anim_info = save_pose_animation(
            out_dir,
            model,
            betas,
            faces,
            anim_weight_scalar,
            joint_id=anim_joint_id,
            axis=args.anim_axis,
            max_angle_deg=args.anim_angle,
            num_frames=args.anim_frames,
            device=device,
            dtype=dtype,
            fps=args.anim_fps,
        )

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("===== SMPL LBS Lab Summary =====\n")
        file.write(f"model_path: {model_path}\n")
        file.write(f"num_vertices: {num_vertices}\n")
        file.write(f"num_faces: {num_faces}\n")
        file.write(f"num_joints_from_lbs_weights: {num_joints}\n")
        file.write(f"num_betas: {args.num_betas}\n")
        file.write(f"visualized_joint_id: {joint_id}\n")
        file.write(f"manual_vs_official_mean_absolute_error: {mean_err:.10f}\n")
        file.write(f"manual_vs_official_max_absolute_error: {max_err:.10f}\n")
        file.write("\nCore intermediate tensors:\n")
        file.write(f"v_template: {tuple(data['v_template'].shape)}\n")
        file.write(f"v_shaped: {tuple(data['v_shaped'].shape)}\n")
        file.write(f"J: {tuple(data['J'].shape)}\n")
        file.write(f"v_posed: {tuple(data['v_posed'].shape)}\n")
        file.write(f"verts: {tuple(data['verts'].shape)}\n")
        if anim_info is not None:
            file.write("\nOptional pose animation:\n")
            file.write(f"anim_joint_id: {args.anim_joint_id}\n")
            file.write(f"anim_axis: {args.anim_axis}\n")
            file.write(f"anim_angle_deg: 0 -> {args.anim_angle}\n")
            file.write(f"anim_num_frames: {anim_info['num_frames']}\n")
            file.write(f"anim_frames_dir: {anim_info['frames_dir']}\n")
            file.write(f"anim_gif: {anim_info['gif_path']}\n")

    print("Work7 SMPL LBS finished.")
    print(f"vertices: {num_vertices}")
    print(f"faces: {num_faces}")
    print(f"joints: {num_joints}")
    print(f"mean absolute error: {mean_err:.10f}")
    print(f"max absolute error: {max_err:.10f}")
    if anim_info is not None:
        print(
            f"animation: joint {args.anim_joint_id} around {args.anim_axis} "
            f"(0 -> {args.anim_angle} deg), {anim_info['num_frames']} frames"
        )
        print(f"animation gif: {anim_info['gif_path']}")
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL LBS visualization lab")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./SMPL_NEUTRAL.pkl",
        help="Path to SMPL_NEUTRAL.pkl",
    )
    parser.add_argument("--out-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--joint-id", type=int, default=18, help="Joint id for weight heatmap")
    parser.add_argument("--num-betas", type=int, default=10, help="Number of shape betas")
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip the optional pose animation (selective content)",
    )
    parser.add_argument(
        "--anim-joint-id",
        type=int,
        default=18,
        help="Joint id rotated in the optional animation (default 18 = left elbow)",
    )
    parser.add_argument(
        "--anim-axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="Local rotation axis for the animated joint",
    )
    parser.add_argument(
        "--anim-angle",
        type=float,
        default=110.0,
        help="Target rotation angle in degrees; the joint sweeps 0 -> this value",
    )
    parser.add_argument(
        "--anim-frames",
        type=int,
        default=24,
        help="Number of animation frames to render",
    )
    parser.add_argument("--anim-fps", type=int, default=12, help="GIF playback frames per second")
    main(parser.parse_args())


