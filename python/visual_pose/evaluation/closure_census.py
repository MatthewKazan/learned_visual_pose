"""
How many of the pose-graph samples' loop closures are wrong, per trajectory, so a training or
val set can be built with or without them. CPU only; reads five small arrays of each sample.npz.

    .venv/bin/python -m visual_pose.evaluation.closure_census

Error of an edge: E = (gt_i^-1 gt_j)^-1 Z_ij, rot = angle(E), trans = |t_E| in metres. sample.npz
stores Z raw (TRANSLATION_NORM is the tokenizer's, applied later). Odometry gets the same
measurement as a convention check: wrong direction/inverse shows as metre-scale odometry error.

    census = load_census()
    PoseGraphDataset(trajectories(census, "training", max_bad=0))

out/closure_census.json: per-trajectory counts at the provisional threshold, small.
out/closure_census_closures.json: every closure's raw errors, to re-apply any threshold.
"""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from roma import rotmat_to_rotvec

from visual_pose.config.constants import REPO_DIR
from visual_pose.data_utils.loaders import split_trajectories
from visual_pose.data_utils.pose_graph_samples import SAMPLE_FILE
from visual_pose.geometry.graph_math import relative

CENSUS_PATH = REPO_DIR / "out" / "closure_census.json"
CLOSURES_PATH = REPO_DIR / "out" / "closure_census_closures.json"
BAD_ROT_DEG = 5.0      # PROVISIONAL: Matt has not chosen tau
BAD_TRANS_M = 0.25


def edge_errors(gt: np.ndarray, ij: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rot_deg, trans_m) per edge of Z (E,4,4) against the gt relative pose, through graph_math.relative."""
    gt_t = torch.from_numpy(gt)[None]
    R_true, t_true = relative(gt_t[..., :3, :3], gt_t[..., :3, 3], torch.from_numpy(ij)[None])
    Z = torch.from_numpy(Z)
    R_err = R_true[0].transpose(-1, -2) @ Z[:, :3, :3]
    rot = torch.rad2deg(rotmat_to_rotvec(R_err).norm(dim=-1))
    # E's translation is R_true^T (t_Z - t_true); a rotation keeps the norm
    trans = (Z[:, :3, 3] - t_true[0]).norm(dim=-1)
    return rot.numpy(), trans.numpy()


def census_entry(traj: Path, role: str) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """(summary record, per-closure detail, odometry rot_deg, odometry trans_m)."""
    with np.load(traj / SAMPLE_FILE) as data:
        gt, cij, cZ, oij, oZ = (data[k] for k in ("gt_poses", "closures_ij", "closures_Z", "odometry_ij", "odometry_Z"))
    rot, trans = edge_errors(gt, cij, cZ)
    odom_rot, odom_trans = edge_errors(gt, oij, oZ)
    bad = (rot > BAD_ROT_DEG) | (trans > BAD_TRANS_M)
    odom_bad = (odom_rot > BAD_ROT_DEG) | (odom_trans > BAD_TRANS_M)
    path = str(traj.relative_to(REPO_DIR))
    summary = dict(
        path=path, role=role, scene=traj.parents[1].name,
        n_vertices=len(gt), n_closures=len(cij), n_bad=int(bad.sum()),
        # same threshold on odometry: the graph's backbone, not filtered by the closure flag
        n_odometry=len(oij), n_odometry_bad=int(odom_bad.sum()),
        odometry_max_rot_deg=round(float(odom_rot.max(initial=0)), 3),
        odometry_max_trans_m=round(float(odom_trans.max(initial=0)), 3),
    )
    # columns, not one dict per closure: 59 MB -> a fraction
    detail = dict(path=path, i=cij[:, 0].tolist(), j=cij[:, 1].tolist(),
                  rot_deg=np.round(rot, 3).tolist(), trans_m=np.round(trans, 3).tolist(),
                  bad=bad.astype(int).tolist())
    return summary, detail, odom_rot, odom_trans


def load_census(path: Path = CENSUS_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)["trajectories"]


def load_closures(path: Path = CLOSURES_PATH) -> dict[str, dict]:
    """path -> {i, j, rot_deg, trans_m, bad}, parallel lists, one entry per closure in sample.npz order."""
    with open(path) as f:
        return {d["path"]: d for d in json.load(f)["trajectories"]}


def trajectories(census: list[dict], role: str, max_bad: int | None = None, min_bad: int | None = None) -> list[Path]:
    """Trajectory dirs of one role with min_bad <= n_bad <= max_bad: PoseGraphDataset's paths_to_trajectories."""
    return [REPO_DIR / t["path"] for t in census if t["role"] == role
            and (max_bad is None or t["n_bad"] <= max_bad) and (min_bad is None or t["n_bad"] >= min_bad)]


def _git() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO_DIR,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _percentiles(x: np.ndarray) -> str:
    q = np.percentile(x, [50, 75, 90, 95, 99, 100]) if len(x) else np.zeros(6)
    return " ".join(f"{v:9.3f}" for v in q)


if __name__ == "__main__":
    train, val = split_trajectories()
    records, details, odom_rot, odom_trans = [], {}, [], []
    for role, trajs in (("training", train), ("validation", val)):
        for n, traj in enumerate(trajs, 1):
            try:
                rec, detail, r, t = census_entry(traj, role)
            except Exception as e:     # a missing or stale sample should not cost the census
                print(f"FAILED {traj}: {e!r}", flush=True)
                continue
            records.append(rec)
            details[rec["path"]] = detail
            odom_rot.append(r)
            odom_trans.append(t)
            if n % 100 == 0:
                print(f"{role} {n}/{len(trajs)}", flush=True)

    summary = {}
    for role in ("training", "validation"):
        rs = [r for r in records if r["role"] == role]
        n_cl, n_bad = sum(r["n_closures"] for r in rs), sum(r["n_bad"] for r in rs)
        summary[role] = dict(n_trajectories=len(rs), n_closures=n_cl, n_bad=n_bad,
                             frac_bad=round(n_bad / max(n_cl, 1), 4),
                             n_trajectories_with_bad=sum(r["n_bad"] > 0 for r in rs),
                             n_trajectories_with_bad_odometry=sum(r["n_odometry_bad"] > 0 for r in rs))
    records.sort(key=lambda r: (r["role"] != "training", -r["n_bad"], r["path"]))
    CENSUS_PATH.parent.mkdir(exist_ok=True)
    header = dict(generated=datetime.date.today().isoformat(), git=_git(),
                  provisional_threshold=dict(rot_deg=BAD_ROT_DEG, trans_m=BAD_TRANS_M,
                                             note="provisional, not chosen by Matt; bad = rot > rot_deg OR trans > trans_m"))
    with open(CENSUS_PATH, "w") as f:
        json.dump(dict(**header, closures_file=CLOSURES_PATH.name, summary=summary, trajectories=records), f, indent=1)
    # one trajectory a line: readable in an editor, and valid JSON
    with open(CLOSURES_PATH, "w") as f:
        f.write(json.dumps(header)[:-1] + ', "trajectories": [\n')
        f.write(",\n".join(json.dumps(details[r["path"]]) for r in records))
        f.write("\n]}\n")

    odom_rot, odom_trans = np.concatenate(odom_rot), np.concatenate(odom_trans)
    print(f"\n{'':24s}{'p50':>9s} {'p75':>9s} {'p90':>9s} {'p95':>9s} {'p99':>9s} {'max':>9s}")
    print(f"{'odometry rot deg':24s}{_percentiles(odom_rot)}")
    print(f"{'odometry trans m':24s}{_percentiles(odom_trans)}")
    odom_bad = (odom_rot > BAD_ROT_DEG) | (odom_trans > BAD_TRANS_M)
    print(f"odometry edges over the closure threshold: {odom_bad.sum()} of {len(odom_bad)}")
    for role in ("training", "validation"):
        ds = [details[r["path"]] for r in records if r["role"] == role]
        rot, trans = (np.array([x for d in ds for x in d[k]]) for k in ("rot_deg", "trans_m"))
        print(f"{role + ' closure rot deg':24s}{_percentiles(rot)}")
        print(f"{role + ' closure trans m':24s}{_percentiles(trans)}")
        for tau in (2, 5, 10, 20):
            print(f"  rot > {tau:4} deg: {(rot > tau).sum():6d}  ({(rot > tau).mean():.2%})")
        for tau in (0.1, 0.25, 0.5, 1.0):
            print(f"  trans > {tau:4} m: {(trans > tau).sum():6d}  ({(trans > tau).mean():.2%})")
    print(json.dumps(summary, indent=1))
    for p in (CENSUS_PATH, CLOSURES_PATH):
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
