"""
HM3D trajectory generator: random walks on the navmesh, rendered by habitat-sim.

    .venv/bin/python -m visual_pose.data_utils.hm3d_renderer                 # every scene in the active splits, TRAJ_PER_SCENE each
    .venv/bin/python -m visual_pose.data_utils.hm3d_renderer --scenes 00801-HaxA7YrQdEC --per-scene 2 --descriptors

Each trajectory is a directory <scene>/trajectories/trajectory_NN/ holding
    poses.txt          one T_WC per line (16 numbers, row-major), habitat axes, eye height included
    traj_points.txt    the navmesh corners the poses were interpolated from
    meta.json          how it was made: seed, navmesh agent, budgets, generator commit
    rgb/NNNNNN.jpg     RGB of frame NNNNNN of poses.txt, JPEG q95
    depth/NNNNNN.png   its depth, uint16 millimetres, 0 = no return
    descriptors.npy    with --descriptors: SemanticDescriptor output per frame, float16
    graph.g2o          the frontend's unsolved pose graph on the saved frames; graph.npz beside it holds the
                       vertex->frame map, ground truth and candidates (pose_graph/frontend.save_graph)

    --overwrite        write trajectory_00.. in order, replacing each one that exists
    --graph-only       no rendering: add the graph to the trajectories already on disk
"""
import argparse
import json
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import quaternion
import torch

from visual_pose.app.run import COMMON
from visual_pose.checkpoints import load_cnn_model
from visual_pose.config import Config
from visual_pose.data_utils.hm3d_sequence import HM3DSequence, IMAGE_SIZE, POSES_FILE, scene_dir, scene_dirs, simulator_config
from visual_pose.data_utils.constants import DEVICE, code_version
from visual_pose.data_utils.loop_closure_dataset import split_of
from visual_pose.data_utils.sources import HM3DSource, SavedHM3DFrames
from visual_pose.pose_graph.frontend import EdgeSolver, Variant, build, has_graph, save_graph
from visual_pose import _geometry as cpp

TRAJ_PER_SCENE = 10
OVERWRITE = True     # write trajectory_00.. in order, replacing each one that exists; else append after the last
GRAPH_ONLY = True    # skip rendering; build the graph for existing trajectories (those without one, or all with OVERWRITE)
# The frontend that builds each trajectory's graph
GRAPH_VARIANT = COMMON[0]
PATH_ANCHORS = 10
MIN_ANCHOR_SPACING_M = 2.0   # anchors closer than this to an earlier one are redrawn: otherwise legs retrace the same doorway
# Frame spacing is keyframe spacing: every frame is stored and the pipeline
# runs on all of them (val_frame_gap 1). 0.5 m matches the 0.6 m the estimator
# was measured at (0.3 m frames, gap 2). Rotation stays tight because turns
# cost overlap (52 deg/frame broke the frontend). Heading is blended along each
# leg, so the rotation budget sets nearly every frame (0 of 551 at the
# translation limit at 0.5 m / 10 deg). Frames on one 118 m walk, 2026-09-30:
# 5.7 deg -> 926, 10 -> 551, 12.5 -> 462, 13.5 -> ~440.
MAX_ROT_BETWEEN_FRAMES = 0.236   # rad per frame, 13.5 deg
MAX_TRANSLATION_BETWEEN_FRAMES = 0.60
EYE_HEIGHT_M = 1.5   # camera above the navmesh point; the sensor itself sits at the agent (hm3d_sequence.simulator_config)
# Navmesh agent, rebuilt before planning. HM3D ships 0.10 m / 1.50 m: paths
# cut corners 10 cm from walls and the eye sat at the ceiling clearance, so
# 2-3% of frames were within 15 cm of geometry and the frontend matched blank
# wall (2026-09-29). 0.30 / 1.80 gives 0% but drops the stairs, so the camera
# never leaves the ground floor; 0.23 keeps them (2026-09-30). Above 0.30
# houses fragment into islands too small to walk. Radius choice per scene:
# SceneRenderer.__init__.
NAVMESH_RADII = (0.23, 0.20, 0.17, 0.14, 0.12, 0.10)
NAVMESH_AGENT_HEIGHT = 1.80
# Wall fill: more than MAX_CLOSE_FRACTION of a frame's pixels within
# MIN_FRAME_DEPTH_M. Listed in meta.json (near_wall.frames); the walk is redrawn
# only when more than MAX_NEAR_WALL_FRACTION of its frames are. A per-frame
# redraw fought the per-scene radius: at 0.14 on 00803's narrow stairs it
# rejected 16 walks in a row, 133 s for one trajectory (2026-10-01).
MIN_FRAME_DEPTH_M = 0.15
MAX_CLOSE_FRACTION = 0.20
MAX_NEAR_WALL_FRACTION = 0.05
# Stop adding anchors once the walk is this long: ten anchors on a 190 m2
# island gave 400 m walks, four times the small scenes' and 400 MB each.
MAX_PATH_M = 150.0
# RGB as JPEG at this quality: 79 KB a frame against 320 KB as PNG (HM3D
# textures are noisy and PNG barely compresses them), mean pixel error 1.4/255.
# Depth stays lossless: uint16 millimetres in PNG, 177 KB.
JPEG_QUALITY = 95
# Height slack for SceneRenderer.off_navmesh; see _segment_on_navmesh.
NAVIGABLE_Y_SLACK = 0.5

# Coverage: 1 m x 1 m floor cells a path crosses. Shortest paths between
# random anchors reuse the same corridors (2026-09-30: 10 uniform anchors on
# 00803 covered 39 of 190 m2, 93% of frames within 1 m of an earlier one), so
# each anchor is the best of ANCHOR_CANDIDATES by new cells per metre of path.
# find_path is ~1 ms; planning a trajectory is well under a second.
COVERAGE_CELL_M = 1.0
ANCHOR_CANDIDATES = 30

def habitat_pose(T: np.ndarray) -> tuple[np.ndarray, quaternion.quaternion]:
    """(position, quaternion) for the agent state from one 4x4 T_WC, Habitat axes."""
    return T[:3, 3], quaternion.from_rotation_matrix(T[:3, :3])


def _segment_on_navmesh(pf, a, b, step: float = 0.1) -> bool:
    """Every sample along a-b is on the navmesh.

    Consecutive poses of a path walked on the navmesh are joined by a
    straight segment lying inside it, so this passes for real legs and
    fails for jumps that were never path-planned.

    Height tolerance is the segment's own rise plus NAVIGABLE_Y_SLACK: a
    staircase is not a straight ramp, so linear y between its corners can
    be a metre off the steps (fixture on 00800, corners 6-7). Same-floor
    segments keep the tight tolerance. Limitation: a wall jump that also
    changes floor gets the loose tolerance and may pass.
    """
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    y_slack = abs(b[1] - a[1]) + NAVIGABLE_Y_SLACK
    return all(pf.is_navigable(a + t * (b - a), y_slack) for t in np.linspace(0, 1, n))


def generate_intermediary_frames(pose1: cpp.PoseSE3, pose2: cpp.PoseSE3, num_frames: int = 10) -> list[cpp.PoseSE3]:
    """num_frames + 1 poses from pose1 to pose2, both endpoints included (so callers slice [1:])."""
    T1 = pose1.matrix()
    T2 = pose2.matrix()

    t1 = T1[:3, 3]
    t2 = T2[:3, 3]

    R1 = cpp.RotationSO3(T1[:3, :3])
    R2 = cpp.RotationSO3(T2[:3, :3])

    relative_R = R1.inverse() * R2
    rot_delta = relative_R.log()

    poses = []

    for i in range(num_frames + 1):
        alpha = i / num_frames

        t = (1.0 - alpha) * t1 + alpha * t2

        # Geodesic rotation interpolation
        R = R1 * cpp.RotationSO3.exp(alpha * rot_delta)

        T = np.eye(4)
        T[:3, :3] = R.matrix()
        T[:3, 3] = t

        poses.append(cpp.PoseSE3(T))

    return poses


def pose_matrix_from_position_heading(p: np.ndarray, h: np.ndarray) -> np.ndarray:
    """4x4 T_WC, Habitat axes, looking along h from navmesh point p; ADDS EYE_HEIGHT_M to p's height."""
    p = np.asarray(p, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    forward = h / np.linalg.norm(h)

    # Habitat camera convention: camera looks along local -Z
    z_axis = -forward

    # Keep camera approximately upright in world Y-up
    world_up = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # cross product of unit vectors in unit
    y_axis = np.cross(z_axis, x_axis)

    T_WC = np.eye(4)
    T_WC[:3, 0] = x_axis
    T_WC[:3, 1] = y_axis
    T_WC[:3, 2] = z_axis
    T_WC[:3, 3] = p

    T_WC[1, 3] += EYE_HEIGHT_M

    return T_WC


def path_cells(points: Sequence[np.ndarray], step: float = 0.5) -> set[tuple[int, int]]:
    """1 m XZ cells crossed by a polyline of navmesh corners, sampled every `step` m."""
    cells = set()
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for t in np.linspace(0.0, 1.0, n + 1):
            p = a + t * (b - a)
            cells.add((int(np.floor(p[0] / COVERAGE_CELL_M)), int(np.floor(p[2] / COVERAGE_CELL_M))))
    return cells


def new_traj_dir(scene_path: Path, out_root: Path | None = None, index: int | None = None) -> Path:
    """
    Create and return trajectory_NN under <scene>/trajectories (or under
    out_root): NN = index, replacing whatever is there, or the next free one.
    Call once per trajectory; every saver writes into it.
    """
    root = (out_root or scene_path) / "trajectories"
    root.mkdir(parents=True, exist_ok=True)
    if index is None:
        taken = [int(p.name.split("_")[-1]) for p in root.iterdir() if p.is_dir() and p.name.split("_")[-1].isdigit()]
        index = max(taken, default=-1) + 1
    path = root / f"trajectory_{index:02d}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    return path


class RawTrajectory:
    """A planned walk, not yet rendered: navmesh corners and the camera poses interpolated along them."""
    def __init__(self, points: Sequence[np.ndarray], coverage_cells: int | None = None):
        self.points = points
        self.coverage_cells = coverage_cells   # 1 m cells the walk crossed; None if not counted
        self.poses = self.generate_traj_poses()

    def generate_traj_poses(self) -> list[cpp.PoseSE3]:
        """T_WC per frame: each corner faces the next, with frames between so no step exceeds the frame budget."""
        positions, headings = [], []
        for a, b in zip(self.points[:-1], self.points[1:]):
            heading = b - a
            heading = heading / np.linalg.norm(heading) if np.linalg.norm(heading) > 1e-6 else headings[-1]
            positions.append(a)
            headings.append(heading)
        positions.append(self.points[-1])   # last corner keeps the heading it arrived with
        headings.append(headings[-1])
        poses = []
        for p, h in zip(positions, headings):
            pose = cpp.PoseSE3(pose_matrix_from_position_heading(p, h))
            if not poses:
                poses.append(pose)
                continue

            diff = pose.inverse() * poses[-1]
            # smallest frame count that keeps both steps under their budget
            nums_frames = max(1,
                              int(np.ceil(np.linalg.norm(diff.translation()) / MAX_TRANSLATION_BETWEEN_FRAMES)),
                              int(np.ceil(diff.rotation().magnitude() / MAX_ROT_BETWEEN_FRAMES)))
            # [1:]: the interpolation starts at poses[-1], which is already there
            poses.extend(generate_intermediary_frames(poses[-1], pose, nums_frames)[1:])
        return poses

    def save_trajectory_poses(self, traj_dir: Path):
        """poses.txt: one pose per line, 16 entries of T_WC row-major, habitat axes."""
        # 12 digits: at 9 the rotation blocks fail PoseSE3's SO(3) check when read back
        np.savetxt(traj_dir / "poses.txt", np.stack([p.matrix().reshape(16) for p in self.poses]), fmt="%.12g")

    def save_trajectory_anchor_points(self, traj_dir: Path):
        """traj_points.txt: the navmesh corners the poses were generated from."""
        np.savetxt(traj_dir / "traj_points.txt", np.asarray(self.points), fmt="%.9g")


class SceneRenderer:
    """One HM3D scene in habitat-sim, on a navmesh rebuilt for the generator's agent: plans walks, renders RGB-D."""
    def __init__(self, scene_dir: Path, seed: int = 0):
        """
        Navmesh radius: the largest in NAVMESH_RADII whose main island still
        spans the full height reached at the smallest radius, within 0.5 m.
        00803's upper stair is narrower than 0.46 m, so 0.23 there loses two levels.
        """
        self.scene_dir = Path(scene_dir)
        self.sim = habitat_sim.Simulator(simulator_config(self.scene_dir))
        self.seed(seed)
        self.agent = self.sim.initialize_agent(0)
        full_span = self._rebuild_navmesh(NAVMESH_RADII[-1])
        for radius in NAVMESH_RADII:
            if self._rebuild_navmesh(radius) >= full_span - 0.5:
                break
        else:
            radius = NAVMESH_RADII[-1]
        self.navmesh_radius = radius

    def _rebuild_navmesh(self, radius: float) -> float:
        """Rebuild for `radius`, pick the largest island, return its height span in metres."""
        navmesh = habitat_sim.NavMeshSettings()
        navmesh.set_defaults()
        navmesh.agent_radius = radius
        navmesh.agent_height = NAVMESH_AGENT_HEIGHT
        if not self.sim.recompute_navmesh(self.sim.pathfinder, navmesh):
            raise RuntimeError(f"navmesh rebuild failed for {self.scene_dir}")
        pf = self.sim.pathfinder
        self.largest_island_index = max(range(pf.num_islands), key=pf.island_area)
        ys = [pf.get_random_navigable_point(island_index=self.largest_island_index)[1] for _ in range(3000)]   # small top levels need many draws to show up
        return max(ys) - min(ys)

    def seed(self, seed: int) -> None:
        """Seeds the navmesh sampler. One seed per trajectory (meta.json) makes each one reproducible on its own."""
        self.seed_value = seed
        self.sim.seed(seed)

    def close(self) -> None:
        self.sim.close()

    def get_random_trajectory(self, num_anchors: int) -> RawTrajectory:
        """
        A walk of num_anchors legs. Each leg's end is the candidate whose
        shortest path from the current anchor crosses the most floor cells
        not yet visited (ties and near-ties broken randomly), at least
        MIN_ANCHOR_SPACING_M from every earlier anchor. Returns the
        RawTrajectory; .coverage_cells is the number of 1 m cells the walk crossed.
        """
        pf = self.sim.pathfinder
        rng = np.random.default_rng(self.seed_value)
        anchors = [pf.get_random_navigable_point(island_index=self.largest_island_index)]
        full_path = [anchors[0]]
        visited = path_cells([anchors[0], anchors[0]])
        length = 0.0
        while len(anchors) < num_anchors and length < MAX_PATH_M:
            best = None
            for _ in range(ANCHOR_CANDIDATES):
                p = pf.get_random_navigable_point(island_index=self.largest_island_index)
                if any(np.linalg.norm(p - q) < MIN_ANCHOR_SPACING_M for q in anchors):
                    continue
                path = habitat_sim.ShortestPath()
                path.requested_start, path.requested_end = anchors[-1], p
                if not pf.find_path(path) or len(path.points) < 2:
                    continue
                cells = path_cells(path.points)
                p_len = sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(path.points[:-1], path.points[1:]))
                score = len(cells - visited) / p_len + rng.uniform(0.0, .01)   # the noise breaks ties between equally new paths
                if best is None or score > best[0]:
                    best = (score, p, list(path.points), cells, p_len)
            if best is None:
                break                                                     # island exhausted at this spacing
            _, p, points, cells, p_len = best
            anchors.append(p)
            full_path.extend(points[1:])                                  # legs share their junction corner
            visited |= cells
            length += p_len
        return RawTrajectory(full_path, coverage_cells=len(visited))

    def render_frame(self, position: np.ndarray, rotation: quaternion.quaternion) -> tuple[np.ndarray, np.ndarray]:
        """Agent state = camera pose. RGB (H, W, 3) uint8 and planar depth (H, W) float32 metres."""
        state = self.agent.get_state()
        state.position = position
        state.rotation = rotation
        self.agent.set_state(state)
        obs = self.sim.get_sensor_observations()
        return obs["rgb"][..., :3], obs["depth"]

    def off_navmesh(self, p: np.ndarray) -> list[bool]:
        """Per consecutive pose pair: does the straight segment leave the navmesh?
        The rebuilt one the generator planned on, not the file HM3D ships."""
        pf = self.sim.pathfinder
        # Poses are at camera height. Drop them to the navmesh by the camera's
        # height above it, so the wall check keeps its tight tolerance. Median
        # over all poses: a single snap can land on a stair landing or a
        # neighbouring floor (00801 trajectory_11, first pose snapped 0.9 m off).
        above = p[:, 1] - np.array([pf.snap_point(q) for q in p])[:, 1]
        above = above[np.isfinite(above)]
        lift = np.array([0.0, float(np.median(above)) if len(above) else 0.0, 0.0])
        return [not _segment_on_navmesh(pf, a - lift, b - lift) for a, b in zip(p[:-1], p[1:])]

    def render_trajectory(self, traj: RawTrajectory) -> HM3DSequence:
        return self.render_poses(np.stack([p.matrix() for p in traj.poses]))

    def render_poses(self, poses: np.ndarray) -> HM3DSequence:
        """poses (N, 4, 4) T_WC in habitat axes, e.g. from a saved poses.txt."""
        rgbs = depths = None
        for i, T in enumerate(poses):
            rgb, depth = self.render_frame(*habitat_pose(T))
            if rgbs is None:
                rgbs = np.empty((len(poses),) + rgb.shape, rgb.dtype)
                depths = np.empty((len(poses),) + depth.shape, np.float32)
            rgbs[i], depths[i] = rgb, depth
        return HM3DSequence(rgbs, depths, poses)


def near_wall_frames(seq: HM3DSequence) -> list[int]:
    """Frames with more than MAX_CLOSE_FRACTION of their valid pixels within MIN_FRAME_DEPTH_M."""
    out = []
    for i in range(len(seq)):
        d = seq.depth(i)
        valid = d > 0
        if valid.any() and np.count_nonzero(valid & (d < MIN_FRAME_DEPTH_M)) > MAX_CLOSE_FRACTION * np.count_nonzero(valid):
            out.append(i)
    return out


def save_meta(traj_dir: Path, renderer: SceneRenderer, traj: RawTrajectory, redraws: int, near_wall: list[int]) -> None:
    """meta.json: everything that decided this trajectory, so it can be regenerated or excluded later."""
    meta = dict(
        scene=renderer.scene_dir.name, seed=renderer.seed_value, redraws=redraws,
        anchors=PATH_ANCHORS, min_anchor_spacing_m=MIN_ANCHOR_SPACING_M,
        navmesh=dict(agent_radius=renderer.navmesh_radius, agent_height=NAVMESH_AGENT_HEIGHT,
                     island=renderer.largest_island_index,
                     island_area_m2=round(float(renderer.sim.pathfinder.island_area(renderer.largest_island_index)), 2)),
        frame_budget=dict(max_translation_m=MAX_TRANSLATION_BETWEEN_FRAMES, max_rotation_rad=MAX_ROT_BETWEEN_FRAMES),
        eye_height_m=EYE_HEIGHT_M, image_size=list(IMAGE_SIZE),
        near_wall=dict(min_depth_m=MIN_FRAME_DEPTH_M, close_fraction=MAX_CLOSE_FRACTION, frames=near_wall),
        poses=len(traj.poses), corners=len(traj.points),
        coverage_m2=traj.coverage_cells,   # 1 m cells the walk crossed; island_area_m2 is the ceiling
        frames=dict(rgb=f"jpeg q{JPEG_QUALITY}", depth="uint16 mm png"),
        generator=code_version(), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (traj_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def save_frames(traj_dir: Path, seq: HM3DSequence) -> None:
    """
    rgb/NNNNNN.jpg and depth/NNNNNN.png for every frame, NNNNNN its row in
    poses.txt. Depth as uint16 millimetres (1 mm quantisation, 0 stays the
    no-return code); RGB as JPEG. See JPEG_QUALITY for sizes.
    """
    (traj_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (traj_dir / "depth").mkdir(parents=True, exist_ok=True)
    for i in range(len(seq)):
        cv2.imwrite(str(traj_dir / "rgb" / f"{i:06d}.jpg"), cv2.cvtColor(seq.rgb(i), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        depth_mm = np.clip(np.round(seq.depth(i) * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(traj_dir / "depth" / f"{i:06d}.png"), depth_mm)


def save_descriptors(traj_dir: Path, seq: HM3DSequence, descriptor: Callable[[torch.Tensor], torch.Tensor],
                     batch_size: int = 128) -> Path:
    """descriptors.npy next to poses.txt: (N, ...) float16, one row per frame, descriptor(x) output order."""
    rgb = np.stack([seq.rgb(i) for i in range(len(seq))])          # (N, H, W, 3) uint8: 1 GB, not 4
    out = []
    with torch.inference_mode():
        for b in range(0, len(rgb), batch_size):
            x = torch.from_numpy(rgb[b:b + batch_size]).to(DEVICE).permute(0, 3, 1, 2).float().div_(255)
            out.append(descriptor(x).to(torch.float16).cpu())
    path = traj_dir / "descriptors.npy"
    np.save(path, torch.cat(out).numpy())
    return path


def generate(renderer: SceneRenderer, seed: int, descriptor: Callable[[torch.Tensor], torch.Tensor] | None = None,
             out_root: Path | None = None, model=None, index: int | None = None) -> Path:
    """
    One trajectory: draw, render, redraw while any frame is a wall fill, save;
    with `model`, its graph too. Written to trajectory_<index>, replacing it,
    or appended after the last. Returns its directory.
    """
    redraws = 0
    while True:
        renderer.seed(seed + 1000 * redraws)
        traj = renderer.get_random_trajectory(PATH_ANCHORS)
        seq = renderer.render_trajectory(traj)
        near_wall = near_wall_frames(seq)
        if len(near_wall) <= MAX_NEAR_WALL_FRACTION * len(seq):
            break
        redraws += 1
    traj_dir = new_traj_dir(renderer.scene_dir, out_root, index)
    traj.save_trajectory_anchor_points(traj_dir)
    traj.save_trajectory_poses(traj_dir)
    save_frames(traj_dir, seq)
    save_meta(traj_dir, renderer, traj, redraws, near_wall)
    if descriptor is not None:
        save_descriptors(traj_dir, seq, descriptor)
    if model is not None:
        build_graph(traj_dir, model)
    return traj_dir


def build_graph(traj_dir: Path, model) -> dict:
    """
    graph.g2o + graph.npz for one trajectory, from the frames ON DISK rather than the
    render in memory, so --graph-only and a fresh render build the same file
    (JPEG q95 changes pixels by 1.4/255 on average). Summary into meta.json.
    """
    front = build(GRAPH_VARIANT, SavedHM3DFrames(traj_dir), model)
    summary = save_graph(traj_dir, front, GRAPH_VARIANT, split_of(f"{HM3DSource.prefix}{traj_dir.parents[1].name}/{traj_dir.name}"))
    meta_path = traj_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["graph"] = summary
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return summary


def graph_model():
    """The descriptor CNN GRAPH_VARIANT names, loaded once per process."""
    model = load_cnn_model(GRAPH_VARIANT.config).eval()
    model.device = next(model.parameters()).device
    return model


def graph_summary(scene: Path, traj_dir: Path, summary: dict) -> str:
    return (f"{scene.name}/{traj_dir.name}: graph {summary['vertices']} vertices, "
            f"{summary['odometry_edges']} odometry edges, {summary['closures']}/{summary['candidates']} closures")


def main(argv: list[str] | None = None) -> None:
    """--per-scene trajectories for each of --scenes, by default every scene in the active HM3D splits."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", help="scene directory names; default: every scene in constants.HM3D_ACTIVE")
    ap.add_argument("--per-scene", type=int, default=TRAJ_PER_SCENE)
    ap.add_argument("--seed", type=int, default=0, help="trajectory k of scene s gets seed + 100*s + k")
    ap.add_argument("--out", type=Path, help="write under OUT/trajectories instead of the scene directory")
    ap.add_argument("--descriptors", action="store_true", help="also write descriptors.npy (frozen DINOv2, models/dino_v2.py)")
    ap.add_argument("--no-graph", action="store_true", help="render only; no graph")
    ap.add_argument("--overwrite", action="store_true", default=OVERWRITE,
                    help="write trajectory_00..(per-scene - 1), replacing each as it goes (with --graph-only: rebuild graphs that exist)")
    ap.add_argument("--graph-only", action="store_true", default=GRAPH_ONLY,
                    help="no rendering: build the graph for the trajectories on disk that lack one")
    args = ap.parse_args(argv)

    scenes = [scene_dir(n) for n in args.scenes] if args.scenes else scene_dirs()
    descriptor = None
    if args.descriptors and not args.graph_only:
        from visual_pose.models.dino_v2 import SemanticDescriptor   # torch.hub download and GPU: only when asked
        descriptor = SemanticDescriptor()
    model = None if args.no_graph else graph_model()

    if args.graph_only:
        for scene in scenes:
            root = (args.out / scene.name if args.out else scene) / "trajectories"
            for traj_dir in sorted(d for d in root.glob("trajectory_*") if (d / POSES_FILE).exists()):
                if has_graph(traj_dir) and not args.overwrite:
                    continue
                print(graph_summary(scene, traj_dir, build_graph(traj_dir, model)), flush=True)
        return

    for s, scene in enumerate(scenes):
        renderer = SceneRenderer(scene, seed=args.seed + 100 * s)
        try:
            for k in range(args.per_scene):
                traj_dir = generate(renderer, args.seed + 100 * s + k, descriptor, args.out and args.out / scene.name, model,
                                    index=k if args.overwrite else None)
                meta = json.loads((traj_dir / "meta.json").read_text())
                print(f"{scene.name}/{traj_dir.name}: {meta['poses']} frames, coverage {meta['coverage_m2']} of {meta['navmesh']['island_area_m2']:.0f} m2, "
                      f"near-wall frames {len(meta['near_wall']['frames'])}, redraws {meta['redraws']}"
                      + (f"; graph {meta['graph']['vertices']} vertices, {meta['graph']['closures']}/{meta['graph']['candidates']} closures"
                         if "graph" in meta else ""), flush=True)
        finally:
            renderer.close()   # one GL context at a time


if __name__ == "__main__":
    main()