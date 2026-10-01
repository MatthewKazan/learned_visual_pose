"""
HM3D trajectory generator: random walks on the navmesh, rendered by habitat-sim.

    .venv/bin/python -m visual_pose.data_utils.hm3d_renderer                 # every scene, TRAJ_PER_SCENE each
    .venv/bin/python -m visual_pose.data_utils.hm3d_renderer --scenes 00801-HaxA7YrQdEC --per-scene 2 --no-descriptors

Each trajectory is a directory <scene>/trajectories/trajectory_NN/ holding
    poses.txt          one T_WC per line (16 numbers, row-major), habitat axes, eye height included
    traj_points.txt    the navmesh corners the poses were interpolated from
    meta.json          how it was made: seed, navmesh agent, budgets, generator commit
    rgb/NNNNNN.jpg     RGB of frame NNNNNN of poses.txt, JPEG q95
    depth/NNNNNN.png   its depth, uint16 millimetres, 0 = no return
    descriptors.npy    optional, SemanticDescriptor output per frame, float16
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import quaternion
import torch

from visual_pose.data_utils.hm3d_sequence import HM3D, HM3DSequence, IMAGE_SIZE, simulator_config
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose import _geometry as cpp


TRAJ_PER_SCENE = 10
PATH_ANCHORS = 10
MIN_ANCHOR_SPACING_M = 2.0   # anchors closer than this to an earlier one are redrawn: otherwise legs retrace the same doorway
# Frame spacing IS the keyframe spacing: every generated frame is stored and
# the pipeline runs on all of them (val_frame_gap 1 for HM3D). 0.5 m matches
# the 0.6 m the estimator was measured at on 0.3 m frames at gap 2. Rotation
# stays tight: overlap is what turns cost (52 deg/frame broke the frontend on
# day one), so turns get extra frames and straight runs do not. Heading is
# blended along each leg, so the ROTATION budget decides nearly every frame
# (0 of 551 frames sat at the translation limit at 0.5 m / 10 deg); the frame
# count scales with it: 5.7 deg -> 926, 10 deg -> 551, 12.5 deg -> 462, 13.5 deg -> ~440 on the
# same 118 m walk (2026-09-30).
MAX_ROT_BETWEEN_FRAMES = 0.236   # rad per frame, 13.5 deg
MAX_TRANSLATION_BETWEEN_FRAMES = 0.60
EYE_HEIGHT_M = 1.5   # camera above the navmesh point; the sensor itself sits at the agent (hm3d_sequence.simulator_config)
# The navmesh is rebuilt for this agent before planning. HM3D ships radius
# 0.10 m, height 1.50 m: paths cut corners 10 cm from walls and the eye sits at
# the ceiling clearance, so 2-3% of frames were within 15 cm of geometry and
# the frontend matched blank wall (2026-09-29). At 0.30 / 1.80 that is 0%,
# but the stairs drop out of the navmesh and the camera never leaves the
# ground floor; 0.23 keeps the staircases (2026-09-30). Above 0.30 the houses
# fragment into islands too small to walk.
# Per scene: the largest radius in NAVMESH_RADII whose main island still spans
# the scene's full height (what radius 0.10 reaches, within 0.5 m). 00803's
# upper stair is narrower than 0.46 m, so 0.23 there loses two levels.
NAVMESH_RADII = (0.23, 0.20, 0.17, 0.14, 0.12, 0.10)
NAVMESH_AGENT_HEIGHT = 1.80
# A frame with more than MAX_CLOSE_FRACTION of its pixels within
# MIN_FRAME_DEPTH_M is a wall fill. Such frames are listed in meta.json as
# near_wall_frames; the walk is only redrawn when more than
# MAX_NEAR_WALL_FRACTION of its frames are like that. A per-frame redraw fought
# the per-scene radius: at 0.14 on 00803's narrow stairs it rejected 16 walks
# in a row, 133 s of rendering for one trajectory (2026-10-01).
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

DESCRIPTOR = None

# Coverage: the walk is scored by the 1 m x 1 m floor cells its path crosses.
# Shortest paths between random anchors reuse a house's corridors whatever the
# anchors are (measured 2026-09-30: 10 uniform anchors on 00803 covered 39 of
# 190 m2 and 93% of frames passed within 1 m of an earlier frame), so each
# anchor is chosen among ANCHOR_CANDIDATES by how many new cells the path to
# it adds. find_path is ~1 ms; planning a trajectory is well under a second.
COVERAGE_CELL_M = 1.0
ANCHOR_CANDIDATES = 30

def habitat_pose(T: np.ndarray):
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


def generate_intermediary_frames(pose1, pose2, num_frames=10):
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

        # Straight-line translation
        t = (1.0 - alpha) * t1 + alpha * t2

        # Geodesic rotation interpolation
        R = R1 * cpp.RotationSO3.exp(alpha * rot_delta)

        T = np.eye(4)
        T[:3, :3] = R.matrix()
        T[:3, 3] = t

        poses.append(cpp.PoseSE3(T))

    return poses


def pose_matrix_from_position_heading(p, h):
    p = np.asarray(p, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    forward = h / np.linalg.norm(h)

    # Habitat camera convention: camera looks along local -Z
    z_axis = -forward

    # Keep camera approximately upright in world Y-up
    world_up = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    T_WC = np.eye(4)
    T_WC[:3, 0] = x_axis
    T_WC[:3, 1] = y_axis
    T_WC[:3, 2] = z_axis
    T_WC[:3, 3] = p

    T_WC[1, 3] += EYE_HEIGHT_M

    return T_WC


def path_cells(points, step: float = 0.5) -> set:
    """1 m XZ cells crossed by a polyline of navmesh corners, sampled every `step` m."""
    cells = set()
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for t in np.linspace(0.0, 1.0, n + 1):
            p = a + t * (b - a)
            cells.add((int(np.floor(p[0] / COVERAGE_CELL_M)), int(np.floor(p[2] / COVERAGE_CELL_M))))
    return cells


def new_traj_dir(scene_path: Path, out_root: Path | None = None) -> Path:
    """Create and return the next trajectory_NN under <scene>/trajectories (or under out_root). Call once per trajectory; every saver writes into it."""
    root = (out_root or scene_path) / "trajectories"
    root.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name.split("_")[1]) for p in root.iterdir() if p.is_dir() and p.name.split("_")[-1].isdigit()]
    path = root / f"trajectory_{max(taken, default=-1) + 1:02d}"
    path.mkdir()
    return path


class RawTrajectory:
    def __init__(self, points):
        self.points = points
        self.poses = self.generate_traj_poses()

    def generate_traj_poses(self):
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
            if nums_frames > 1:
                # [1:]: the interpolation starts at poses[-1], which is already there
                poses.extend(generate_intermediary_frames(poses[-1], pose, nums_frames)[1:])
            else:
                poses.append(pose)
        return poses

    def save_trajectory_poses(self, traj_dir: Path):
        """poses.txt: one pose per line, 16 entries of T_WC row-major, habitat axes."""
        # 12 digits: at 9 the rotation blocks fail PoseSE3's SO(3) check when read back
        np.savetxt(traj_dir / "poses.txt", np.stack([p.matrix().reshape(16) for p in self.poses]), fmt="%.12g")

    def save_trajectory_anchor_points(self, traj_dir: Path):
        """traj_points.txt: the navmesh corners the poses were generated from."""
        np.savetxt(traj_dir / "traj_points.txt", np.asarray(self.points), fmt="%.9g")


class SceneRenderer:
    def __init__(self, scene_dir: Path, seed: int = 0):
        self.scene_dir = Path(scene_dir)
        self.sim = habitat_sim.Simulator(simulator_config(self.scene_dir))
        self.seed(seed)
        self.agent = self.sim.initialize_agent(0)
        full_span = self._rebuild_navmesh(NAVMESH_RADII[-1])
        chosen_radius = 0.1
        for radius in NAVMESH_RADII:
            if self._rebuild_navmesh(radius) >= full_span - 0.5:
                chosen_radius = radius
                break
        self.navmesh_radius = chosen_radius

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
        np.random.seed(seed)

    def close(self) -> None:
        self.sim.close()

    def get_random_trajectory(self, num_anchors: int):
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
                    best = (score, p, list(path.points), cells)
            if best is None:
                break                                                     # island exhausted at this spacing
            _, p, points, cells = best
            anchors.append(p)
            full_path.extend(points[1:])                                  # legs share their junction corner
            visited |= cells
            length += sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(points[:-1], points[1:]))
        traj = RawTrajectory(full_path)
        traj.coverage_cells = len(visited)
        return traj

    def render_frame(self, position, rotation):
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


def generator_version() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_DIR), "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:   # noqa: BLE001 -- provenance must never abort generation
        return "unknown"


def save_meta(traj_dir: Path, renderer: "SceneRenderer", traj: RawTrajectory, redraws: int, near_wall: list[int]) -> None:
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
        coverage_m2=getattr(traj, "coverage_cells", None),   # 1 m cells the walk crossed; island_area_m2 is the ceiling
        frames=dict(rgb=f"jpeg q{JPEG_QUALITY}", depth="uint16 mm png"),
        generator=generator_version(), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
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


def save_descriptors(traj_dir: Path, traj_seq: HM3DSequence, descriptor,
                     batch_size: int = 128) -> Path:
    """descriptors.npy next to poses.txt: (N, ...) float16, one row per frame, descriptor(x) output order."""
    rgb = np.stack([traj_seq.rgb(i) for i in range(len(traj_seq))])          # (N, H, W, 3) uint8: 1 GB, not 4
    out = []
    with torch.inference_mode():
        for b in range(0, len(rgb), batch_size):
            x = torch.from_numpy(rgb[b:b + batch_size]).to(DEVICE).permute(0, 3, 1, 2).float().div_(255)
            out.append(descriptor(x).to(torch.float16).cpu())
    path = traj_dir / "descriptors.npy"
    np.save(path, torch.cat(out).numpy())
    return path


def generate(scene: Path, renderer: SceneRenderer, seed: int, descriptor=None, out_root: Path | None = None) -> Path:
    """One trajectory: draw, render, redraw while any frame is a wall fill, save. Returns its directory."""
    redraws = 0
    while True:
        renderer.seed(seed + 1000 * redraws)
        traj = renderer.get_random_trajectory(PATH_ANCHORS)
        seq = renderer.render_trajectory(traj)
        near_wall = near_wall_frames(seq)
        if len(near_wall) <= MAX_NEAR_WALL_FRACTION * len(seq):
            break
        redraws += 1
    traj_dir = new_traj_dir(scene, out_root)
    traj.save_trajectory_anchor_points(traj_dir)
    traj.save_trajectory_poses(traj_dir)
    save_frames(traj_dir, seq)
    save_meta(traj_dir, renderer, traj, redraws, near_wall)
    if descriptor is not None:
        save_descriptors(traj_dir, seq, descriptor)
    return traj_dir

HM3D = REPO_DIR / "data" / "HM3D" / "hm3d-val-habitat-v0.2"

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", help="scene directory names; default: every scene under data/hm3d")
    ap.add_argument("--per-scene", type=int, default=TRAJ_PER_SCENE)
    ap.add_argument("--seed", type=int, default=0, help="trajectory k of scene s gets seed + 100*s + k")
    ap.add_argument("--out", type=Path, help="write under OUT/trajectories instead of the scene directory")
    args = ap.parse_args(argv)

    scenes = [HM3D / n for n in args.scenes] if args.scenes else sorted(p for p in HM3D.iterdir() if p.is_dir())
    descriptor = DESCRIPTOR

    for s, scene in enumerate(scenes):
        renderer = SceneRenderer(scene, seed=args.seed + 100 * s)
        try:
            for k in range(args.per_scene):
                traj_dir = generate(scene, renderer, args.seed + 100 * s + k, descriptor, args.out and args.out / scene.name)
                meta = json.loads((traj_dir / "meta.json").read_text())
                print(f"{scene.name}/{traj_dir.name}: {meta['poses']} frames, coverage {meta['coverage_m2']} of {meta['navmesh']['island_area_m2']:.0f} m2, "
                      f"near-wall frames {len(meta['near_wall']['frames'])}, redraws {meta['redraws']}", flush=True)
        finally:
            renderer.close()   # one GL context at a time


if __name__ == "__main__":
    main()