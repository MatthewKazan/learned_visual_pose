"""
The console comparison tables -- the numbers that decide things.
"""
from __future__ import annotations

import numpy as np

from visual_pose.evaluation.metrics import chain, drift, path_length


def report(errors: dict, poses: dict, gt: list, metric: set) -> None:
    """Per-estimator percentile table.

    `metric` is the set of row names whose t is in metres, not a bool. The
    estimator column is 20 characters; a longer name shifts that row.
    """
    print(f"\n{'estimator':>20} | {'rotation (deg)':^23} | {'t dir':>6} | {'|t| cm':>7}")
    print(f"{'':>20} | {'p50':>7} {'p90':>7} {'max':>7} | {'p50':>6} | {'p50':>7}")
    for name, err in errors.items():
        e = np.array(err)
        r = np.nanpercentile(e[:, 0], [50, 90])
        mag = f"{np.nanmedian(e[:, 2]):>7.2f}" if name in metric else f"{'-':>7}"
        print(f"{name:>20} | {r[0]:>7.3f} {r[1]:>7.3f} {np.nanmax(e[:, 0]):>7.3f} | "
              f"{np.nanmedian(e[:, 1]):>6.2f} | {mag}")

    # rotation compounds down a chain and translation direction does not, so the
    # tail predicts drift far better than any median
    gt_pos = chain(gt, gt, metric=True)
    length = path_length(gt_pos)      # header only; drift recomputes it per row
    print(f"\n{'estimator':>20} | >5 deg | failed | final err | drift  "
          f"(over {length:.1f} m)")
    for name, err in errors.items():
        e = np.array(err)
        final, pct = drift(chain(poses[name], gt, name in metric), gt_pos)
        print(f"{name:>20} | {int(np.nansum(e[:, 0] > 5)):>6} | "
              f"{int(np.isnan(e[:, 0]).sum()):>6} | {final:>7.2f} m | "
              f"{pct:>5.1f}%")


def report_path(label: str, path: np.ndarray, gt_path: np.ndarray,
                length: float) -> None:
    """One trajectory against ground truth: ATE, drift, and path length.
    """
    final = np.linalg.norm(path[-1] - gt_path[-1])
    error = np.sqrt(np.mean(np.sum((path - gt_path) ** 2, axis=1)))
    print(f"  {label:>28}: final err {final:6.2f} m  ATE {error:6.2f} m  "
          f"drift {100 * final / length:5.1f}%")


def worst_edges(errors: dict, frames: list[int], n: int = 5,
                matches: list[int] | None = None) -> None:
    """The few worst edges per metric, by frame index.
    """
    names = list(errors)
    rot = np.array([[e[0] for e in errors[k]] for k in names])   # (rows, edges)
    order = np.argsort(np.nan_to_num(np.nanmax(rot, axis=0), nan=-1))[::-1][:n]

    w = [max(9, len(k) + 2) for k in names]
    print(f"\nworst {n} edges by rotation error (deg)")
    header = f"{'edge':>13} |" + "".join(f"{k:>{c}}" for k, c in zip(names, w))
    print(header + ("  matches" if matches is not None else ""))
    for idx in order:
        line = f"{frames[idx]:5d}->{frames[idx + 1]:5d} |" + "".join(
            f"{rot[r, idx]:{c}.3f}" for r, c in enumerate(w))
        if matches is not None:
            line += f"{matches[idx]:9d}"
        print(line)
