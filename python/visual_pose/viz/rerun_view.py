"""
What rerun_app logs into each scene's recording. Everything is in the scene
sequence's world frame; predictions (logged in keyframe 0's camera frame) are
placed there as

    anchored   T_WC_0 T_0k                  as plain ATE scores them
    aligned    A T_WC_0 T_0k                A: the rigid fit ate_aligned uses

Entities:
    world/truth/{path, ends, camera, cloud/<frame>}, image, depth
    world/runs/<run>/<track>/<placement>/{path, camera, error_line, cloud/<frame>}
    world/live/<track>, error/<run>/<track>/<placement>, info
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from visual_pose.data_utils.constants import REPO_DIR
from visual_pose.evaluation.metrics import rigid_alignment
from visual_pose.geometry.true_correspondences import back_project, transform_points

PIXEL_STRIDE = 16
CLOUD_EVERY = 2          # for sequences whose frames are not already keyframes
IMAGE_SCALE, DEPTH_SCALE, JPEG_QUALITY = 2, 4, 85
TRUTH_RADIUS, PRED_RADIUS = 0.014, 0.007    # smaller predictions: coinciding clouds hid the truth's colours
CACHE_DIR = REPO_DIR / "out" / "cache" / "rerun"
PRED_STALE_M = 0.05      # logged truth this far off the sequence: the data changed since the run
PRED_COLORS = ("#ff7f0e", "#ffd400", "#2563eb", "#16a34a", "#e11d48", "#7c3aed", "#a16207", "#0f172a")
SCENE_HIDDEN = ["world/truth/anchors", "world/truth/intermediary"]


def viewer_executable() -> str:
    return str(Path(sys.executable).parent / "rerun")     # .venv/bin is not on PATH


def _hex(color: str) -> list[int]:
    return [int(color[i:i + 2], 16) for i in (1, 3, 5)]


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def run_slug(entry) -> str:
    return f"{entry.log.name[4:8]}_{entry.log.name[9:15]}"


def blueprint(hidden: list[str], *, frames: bool):
    import rerun.blueprint as rrb
    behaviour = {path: rrb.EntityBehavior(visible=False) for path in hidden}
    side = [rrb.TextDocumentView(origin="info", name="info")]
    if frames:
        side += [rrb.Spatial2DView(origin="image", name="camera"), rrb.Spatial2DView(origin="depth", name="depth")]
    side.append(rrb.TimeSeriesView(origin="error", name="camera position error (m)", overrides=behaviour))
    return rrb.Blueprint(rrb.Horizontal(rrb.Spatial3DView(origin="world", name="scene", overrides=behaviour),
                                        rrb.Vertical(*side), column_shares=[3, 1]),
                         rrb.TimePanel(timeline="frame", fps=8), rrb.SelectionPanel(state="collapsed"))


def hidden_by_default(entries, has_truth: bool) -> list[str]:
    """Older runs, anchored when aligned exists, and the odometry map when a graph twin does."""
    out = list(SCENE_HIDDEN)
    for i, e in enumerate(sorted(entries, key=lambda e: e.log.name, reverse=True)):
        run = run_slug(e)
        if i:
            out += [f"world/runs/{run}", f"error/{run}"]
            continue
        twin = any(p.derived for p in e.predictions)
        for p in e.predictions:
            if has_truth:
                out += [f"world/runs/{run}/{slug(p.name)}/anchored", f"error/{run}/{slug(p.name)}/anchored"]
            if twin and not p.derived:
                out += [f"world/runs/{run}/{slug(p.name)}/{pl}/cloud" for pl in ("aligned", "anchored")]
    return out


def place(T_true: np.ndarray | None, pred) -> dict:
    """placement -> (points, poses or None) in the world. T_true: true T_WC per frame, or None."""
    T0 = np.eye(4) if T_true is None else T_true[0 if pred.frames is None else pred.frames[0]]
    points = pred.positions @ T0[:3, :3].T + T0[:3, 3]
    poses = None if pred.poses is None else T0 @ pred.poses
    out = {"anchored": (points, poses)}
    if T_true is not None and pred.frames is not None and len(pred.frames) == len(points):
        A = rigid_alignment(points, T_true[pred.frames, :3, 3])
        out["aligned"] = (points @ A[:3, :3].T + A[:3, 3], None if poses is None else A @ poses)
    return out


def is_stale(T_true: np.ndarray, reference: np.ndarray) -> bool:
    if not len(reference):
        return False
    ref = reference @ T_true[0, :3, :3].T + T_true[0, :3, 3]
    return float(np.linalg.norm(ref[:, None] - T_true[None, :, :3, 3], axis=2).min(axis=1).max()) > PRED_STALE_M


def log_scene(rec, axes: str, truth: np.ndarray) -> None:
    """truth: (N, 3) true positions, empty without ground truth."""
    import rerun as rr
    rec.log("world", getattr(rr.ViewCoordinates, axes), static=True)
    if len(truth):
        rec.log("world/truth/path", rr.LineStrips3D([truth], colors=[255, 255, 255], radii=0.02), static=True)
        rec.log("world/truth/ends", rr.Points3D([truth[0], truth[-1]], colors=[[0, 255, 0], [255, 0, 0]],
                                                radii=0.15, labels=["start", "end"]), static=True)


def _pinhole(rec, entity: str, color, K: np.ndarray, shape) -> None:
    import rerun as rr
    rec.log(entity, rr.Pinhole(image_from_camera=K, resolution=[shape[1], shape[0]],
                               camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.3, color=color), static=True)


def log_run(rec, seq, T_true, entry, placed: dict, colors: dict) -> None:
    """Paths; with a sequence, the camera per keyframe; with ground truth, its error."""
    import rerun as rr
    run = run_slug(entry)
    for pred in entry.predictions:
        rgb = _hex(colors[pred.track])
        for placement, (points, poses) in placed[pred.track].items():
            base = f"world/runs/{run}/{slug(pred.name)}/{placement}"
            rec.log(f"{base}/path", rr.LineStrips3D([points], colors=[rgb], radii=0.02), static=True)
            if poses is None:
                continue
            _pinhole(rec, f"{base}/camera", rgb, seq.K, seq.depth(0).shape)
            series = f"error/{run}/{slug(pred.name)}/{placement}"
            rec.log(series, rr.SeriesLines(colors=[rgb], names=[f"{run} {pred.name} {placement}"]), static=True)
            for f, T in zip(pred.frames.tolist(), poses):
                rec.set_time("frame", sequence=f)
                rec.log(f"{base}/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
                if T_true is not None:
                    rec.log(f"{base}/error_line", rr.LineStrips3D([[T_true[f, :3, 3], T[:3, 3]]], colors=[rgb]))
                    rec.log(series, rr.Scalars(float(np.linalg.norm(T[:3, 3] - T_true[f, :3, 3]))))


def log_info(rec, title: str, lines: list[str], entries) -> None:
    import rerun as rr
    rows = ["| run | config | track | ATE |", "|---|---|---|---|"]
    rows += [f"| {e.when} | {e.config} | {p.name} | {p.ate or '-'} |"
             for e in sorted(entries, key=lambda e: e.log.name, reverse=True) for p in e.predictions]
    text = [f"**{title}**", ""] + lines + ["", "older runs and anchored placements start hidden (eye icons)", ""]
    rec.log("info", rr.TextDocument("\n".join(text + (rows if entries else ["no finished runs yet"])),
                                    media_type=rr.MediaType.MARKDOWN), static=True)


# -- frames and maps: sent when a recording is first opened -----------------------

def _grid(H: int, W: int) -> np.ndarray:
    u, v = np.meshgrid(np.arange(PIXEL_STRIDE // 2, W, PIXEL_STRIDE), np.arange(PIXEL_STRIDE // 2, H, PIXEL_STRIDE))
    return np.stack([u.ravel(), v.ravel()], axis=1)


def samples(src, seq) -> dict:
    """Per frame of seq: jpeg, small depth, and the cloud grid's depth and colour; cached by src.stamp()."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path, stamp = CACHE_DIR / f"{slug(src.key)}__s{PIXEL_STRIDE}.npz", src.stamp()
    if path.exists() and float((z := np.load(path))["stamp"]) == stamp:
        blob, ends = z["jpeg"].tobytes(), z["jpeg_ends"]
        return {"jpeg": [blob[a:b] for a, b in zip(np.r_[0, ends[:-1]], ends)],
                "depth": z["depth"], "grid_depth": z["grid_depth"], "grid_rgb": z["grid_rgb"]}
    t0 = time.time()
    uv = _grid(*seq.depth(0).shape)

    def one(i):
        rgb, depth = seq.rgb(i), seq.depth(i)
        jpg = cv2.imencode(".jpg", np.ascontiguousarray(rgb[::IMAGE_SCALE, ::IMAGE_SCALE, ::-1]),
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
        d = depth[::DEPTH_SCALE, ::DEPTH_SCALE]
        return (jpg, np.where(d < src.max_depth, d, 0).astype(np.float16),
                depth[uv[:, 1], uv[:, 0]], rgb[uv[:, 1], uv[:, 0]])

    with ThreadPoolExecutor(8) as pool:            # cv2 decodes without the GIL
        jpeg, depth, grid_depth, grid_rgb = zip(*pool.map(one, range(len(seq))))
    out = {"jpeg": list(jpeg), "depth": np.stack(depth), "grid_depth": np.stack(grid_depth),
           "grid_rgb": np.stack(grid_rgb)}
    np.savez(path, stamp=stamp, jpeg=np.frombuffer(b"".join(jpeg), np.uint8),
             jpeg_ends=np.cumsum([len(j) for j in jpeg]), depth=out["depth"],
             grid_depth=out["grid_depth"], grid_rgb=out["grid_rgb"])
    print(f"  {len(jpeg)} frames of {src.key} in {time.time() - t0:.1f}s")
    return out


def _cloud(r, f, uv, K, max_depth):
    depth = r["grid_depth"][f]
    keep = (depth > 0) & (depth < max_depth)
    return back_project(uv[keep].astype(float), depth[keep].astype(float), K), keep


def log_frames(rec, r: dict, seq, T_true, max_depth: float) -> None:
    """Image and depth at every frame; with ground truth, the true camera and map."""
    import rerun as rr
    uv = _grid(*seq.depth(0).shape)
    step = 1 if seq.frames_are_keyframes else CLOUD_EVERY
    if T_true is not None:
        _pinhole(rec, "world/truth/camera", [255, 255, 255], seq.K, seq.depth(0).shape)
    for f in range(len(seq)):
        rec.set_time("frame", sequence=f)
        rec.log("image", rr.EncodedImage(contents=r["jpeg"][f], media_type="image/jpeg"))
        rec.log("depth", rr.DepthImage(r["depth"][f].astype(np.float32), meter=1.0))
        if T_true is None:
            continue
        rec.log("world/truth/camera", rr.Transform3D(translation=T_true[f, :3, 3], mat3x3=T_true[f, :3, :3]))
        if f % step == 0:
            P, keep = _cloud(r, f, uv, seq.K, max_depth)
            # an entity per keyframe, so the map accumulates as the timeline plays
            rec.log(f"world/truth/cloud/{f:05d}", rr.Points3D(transform_points(P, T_true[f]),
                                                              colors=r["grid_rgb"][f][keep], radii=TRUTH_RADIUS))


def log_run_clouds(rec, r: dict, seq, max_depth: float, entry, placed: dict, colors: dict,
                   own_colors: bool) -> None:
    """own_colors: each point its pixel's colour, for when there is no truth map to tell apart from."""
    import rerun as rr
    uv = _grid(*seq.depth(0).shape)
    for pred in entry.predictions:
        for placement, (_, poses) in placed[pred.track].items():
            if poses is None:
                continue
            base = f"world/runs/{run_slug(entry)}/{slug(pred.name)}/{placement}"
            for f, T in zip(pred.frames.tolist(), poses):
                P, keep = _cloud(r, f, uv, seq.K, max_depth)
                rec.set_time("frame", sequence=f)
                rec.log(f"{base}/cloud/{f:05d}", rr.Points3D(
                    transform_points(P, T), radii=PRED_RADIUS,
                    colors=r["grid_rgb"][f][keep] if own_colors else _hex(colors[pred.track])))


# -- per-source extras: (rec, src, renderer) -> info lines --------------------------

def log_hm3d_extras(rec, src, renderer) -> list[str]:
    """Navmesh (the one SceneRenderer rebuilt and the generator planned on),
    segments leaving it, the generator's corners and interpolated poses."""
    import matplotlib.pyplot as plt
    import rerun as rr
    from visual_pose.data_utils.hm3d_renderer import EYE_HEIGHT_M
    from visual_pose.data_utils.hm3d_sequence import POSES_FILE, load_poses
    pf = renderer.sim.pathfinder
    verts = np.array(pf.build_navmesh_vertices(), dtype=np.float32)
    y = verts[:, 1]
    rec.log("world/navmesh", rr.Mesh3D(
        vertex_positions=verts, triangle_indices=np.array(pf.build_navmesh_vertex_indices()).reshape(-1, 3),
        vertex_colors=(plt.get_cmap("cool")((y - y.min()) / max(np.ptp(y), 1e-6))[:, :3] * 255).astype(np.uint8)),
        static=True)
    p = load_poses(src.dir / POSES_FILE)[:, :3, 3]
    bad = renderer.off_navmesh(p)
    if any(bad):
        rec.log("world/truth/off_navmesh", rr.LineStrips3D(
            [[a, b] for a, b, x in zip(p[:-1], p[1:], bad) if x], colors=[255, 0, 0], radii=0.05), static=True)
    if (src.dir / "traj_points.txt").exists():
        corners = np.loadtxt(src.dir / "traj_points.txt", ndmin=2) + [0.0, EYE_HEIGHT_M, 0.0]
        between = np.flatnonzero(np.linalg.norm(p[:, None] - corners[None], axis=2).min(axis=1) > 1e-4)
        rec.log("world/truth/anchors", rr.Points3D(corners, colors=[255, 255, 255], radii=0.08,
                                                   labels=[f"corner {i}" for i in range(len(corners))]), static=True)
        rec.log("world/truth/intermediary", rr.Points3D(p[between], colors=[160, 160, 160], radii=0.03,
                                                        labels=[f"frame {f}" for f in between]), static=True)
    return [f"**{sum(bad)} segments leave the navmesh**" if any(bad) else "path stays on the navmesh"]
