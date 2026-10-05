"""
Append-only run log: how an estimator talks to the viewer without importing it.

Writer side -- any process, any code, no matplotlib:

    with RunWriter("kabsch g5", meta={"gap": 5}) as run:
        run.reference(gt_path)
        for k, ... in enumerate(edges):
            run.frame(path_so_far, x=k, x_label="edge", **{"rot err (deg)": e})

Reader side: `python -m visual_pose.viz.rerun_app` tails the directory and
draws whatever appears in rerun. Neither side imports the other, which is the point:

  * edit the estimator and re-run it as often as you like -- new process, new
    code, and the window never restarts
  * an estimator that crashes leaves a partial log, not a dead GUI
  * a finished log still plots next week, so runs are comparable across days

One JSON object per line, so a half-written run is a valid run minus its last
line and `tail -f` is a debugger. Cost is roughly 25 bytes per vertex per frame:
120 vertices x 120 frames is ~400 kB, but 1000 vertices x 500 iterations is
~12 MB -- log fewer frames if that ever matters.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from visual_pose.data_utils.constants import REPO_DIR

DEFAULT_ROOT = REPO_DIR / "out" / "runs"

SUFFIX = ".jsonl"


def positions_of(poses) -> np.ndarray:
    """
    Translations as (N, 3), from any of: (N, 4, 4), a list of 4x4, a list of
    PoseSE3, or already-plain (N, 2) / (N, 3) points.
    """
    seq = list(poses)
    if seq and hasattr(seq[0], "translation"):
        return np.array([np.asarray(p.translation(), float).ravel() for p in seq])
    arr = np.asarray(seq, dtype=float)
    if arr.ndim == 3:
        return arr[:, :3, 3].copy()
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.c_[arr, np.zeros(len(arr))]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.copy()
    raise ValueError(f"cannot read positions from shape {arr.shape}")


def _slug(text: str) -> str:
    keep = [c if c.isalnum() or c in "-_" else "-" for c in text.lower()]
    return "".join(keep).strip("-")[:60] or "run"


class RunWriter:
    """
    One run, one file. Flushed on every record so a reader tailing it sees each
    frame as it happens rather than when the OS feels like it.
    """

    def __init__(self, name: str, *, root: Path | str | None = None,
                 meta: Mapping | None = None, decimals: int = 5,
                 path: Path | str | None = None):
        self.name = name
        self.decimals = decimals
        # Timestamp first so the directory sorts chronologically, which is also
        # the order the viewer replays existing runs in.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = Path(path) if path else \
            Path(root or DEFAULT_ROOT) / f"{stamp}_{_slug(name)}{SUFFIX}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", buffering=1)
        self._tracks: list[str] = []
        self._write(kind="meta", name=name, created=time.time(),
                    meta=dict(meta or {}))

    # -- writing -------------------------------------------------------------

    def _write(self, **record) -> None:
        json.dump(record, self._file, separators=(",", ":"))
        self._file.write("\n")
        self._file.flush()

    def _points(self, poses) -> list:
        return np.round(positions_of(poses), self.decimals).tolist()

    def declare_track(self, track: str, *, linestyle: str | None = None,
                      color: str | None = None, show_edges: bool = False,
                      derived_from: str | None = None) -> None:
        """Display hints for one track. `derived_from` reuses that track's colour.
    """
        self._write(kind="track", track=track, linestyle=linestyle, color=color,
                    show_edges=show_edges, derived_from=derived_from)

    def clear_viewer(self) -> None:
        """Ask the viewer to wipe the screen before this run draws.

    A record, not a call: the writer has no handle on the window.
    """
        self._write(kind="clear")

    def reference(self, poses, label: str = "ground truth") -> None:
        self._write(kind="reference", positions=self._points(poses), label=label)

    def frame(self, poses, *, track: str | None = None, x: float | None = None,
              x_label: str = "iteration", edges=None, keep: bool = False,
              **scalars) -> None:
        """One state of one track. `scalars` become series in the viewer."""
        track = track or self.name
        if track not in self._tracks:
            self._tracks.append(track)
        self._write(
            kind="frame", track=track, positions=self._points(poses),
            x=None if x is None else float(x), x_label=x_label,
            scalars={k: float(v) for k, v in scalars.items()
                     if v is not None and np.isfinite(v)},
            edges=None if edges is None else np.asarray(edges, int).tolist(),
            keep=keep,
        )

    def end(self, track: str | None = None) -> None:
        """Freeze a track. Called for every open track on close."""
        self._write(kind="end", track=track or self.name)

    def poses(self, track: str, frames, poses) -> None:
        """A track's final (N, 4, 4) poses, T_0k in the first keyframe's camera
        frame, and the sequence frame each belongs to. Rotation included, which
        `frame` drops: the point-cloud viewer places depth with it."""
        self._write(kind="poses", track=track, frames=[int(f) for f in frames],
                    poses=np.round(np.asarray(poses, float), 6).tolist())

    def closures(self, track: str, pairs, T_ji=None) -> None:
        """Verified loop closures as sequence frame pairs (the frames of `poses`),
        and optionally each closure's measured T_ji."""
        self._write(kind="closures", track=track, pairs=np.asarray(pairs, int).reshape(-1, 2).tolist(),
                    T_ji=None if T_ji is None else np.round(np.asarray(T_ji, float), 6).tolist())

    def summary(self, track: str, text: str) -> None:
        """A one-line result the viewer appends to the track's legend entry."""
        self._write(kind="summary", track=track, text=text)

    def table(self, title: str, text: str) -> None:
        """Preformatted text the viewer shows in its right column."""
        self._write(kind="table", title=title, text=text)

    def note(self, text: str) -> None:
        """A line of console output worth keeping with the run."""
        self._write(kind="note", text=text)

    def close(self) -> None:
        if self._file.closed:
            return
        for track in self._tracks:
            self._write(kind="end", track=track)
        # Absence of this record is how a reader tells "still running" from
        # "the writer died", so it must be the last thing written.
        self._write(kind="closed")
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._write(kind="note", text=f"failed: {exc_type.__name__}: {exc}")
        self.close()


# -- reading -----------------------------------------------------------------

def records(path: Path | str) -> Iterator[dict]:
    """Every record in a finished (or partial) log, once."""
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def tail(path: Path | str, *, stop: Callable[[], bool] | None = None,
         poll: float = 0.15, idle_timeout: float = 900.0) -> Iterator[dict]:
    """Yield records as they are appended; stops when `stop()` returns True.
    """
    path = Path(path)
    pending = ""
    idle_since = time.monotonic()
    with path.open() as handle:
        while True:
            chunk = handle.readline()
            if chunk and chunk.endswith("\n"):
                line = (pending + chunk).strip()
                pending = ""
                idle_since = time.monotonic()
                if line:
                    record = json.loads(line)
                    if record.get("kind") == "closed":
                        return
                    yield record
                continue
            pending += chunk
            if stop is not None and stop():
                return
            if time.monotonic() - idle_since > idle_timeout:
                return
            time.sleep(poll)


def list_runs(root: Path | str = DEFAULT_ROOT) -> list[Path]:
    """Logs oldest first, which is the order they should be replayed in."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.glob(f"*{SUFFIX}"), key=lambda p: os.stat(p).st_mtime)


def summary(path: Path | str) -> dict:
    """Header, track names and frame count, without loading the positions."""
    name, meta, tracks, frames, closed = Path(path).stem, {}, [], 0, False
    for record in records(path):
        kind = record.get("kind")
        if kind == "meta":
            name, meta = record.get("name", name), record.get("meta", {})
        elif kind == "frame":
            frames += 1
            if record["track"] not in tracks:
                tracks.append(record["track"])
        elif kind == "closed":
            closed = True
    return {"name": name, "meta": meta, "tracks": tracks,
            "frames": frames, "closed": closed, "path": Path(path)}


# -- a finished run, as the viewer uses it -----------------------------------

@dataclass
class Prediction:
    log: Path
    track: str
    name: str                 # short: "odometry", "+ gauss newton", ...
    derived: bool             # the graph twin of another track
    positions: np.ndarray     # (N, 3) in keyframe 0's OpenCV camera frame
    ate: str | None
    # From the "poses" record, which logs before 2026-09-30 lack: poses.txt
    # frame of each pose, and T_0k with rotation (the 3D map needs it)
    frames: np.ndarray | None = None
    poses: np.ndarray | None = None
    closures: np.ndarray | None = None    # (C, 2) frame pairs; a graph twin shares its parent's
    closure_T_ji: np.ndarray | None = None  # (C, 4, 4) measured, when logged


@dataclass
class RunEntry:
    log: Path
    when: str                 # "09-29 16:21"
    config: str               # the variant label minus the scene
    predictions: list[Prediction]
    reference: np.ndarray     # the log's ground truth, same frame as positions


def read_run_log(path: Path) -> RunEntry:
    """Last frame of every track, plus its ATE: summary records, or the older
    "ATE a m -> b m" note, which names (odometry, graph) in that order."""
    records = []
    with path.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass              # a run still writing its last line
    meta = records[0]
    scene = meta.get("name", "")
    last, hints, ate, full, noted, closures = {}, {}, {}, {}, None, {}
    reference = np.zeros((0, 3))
    for r in records:
        kind = r.get("kind")
        if kind == "frame":
            last[r["track"]] = r["positions"]
        elif kind == "track":
            hints[r["track"]] = r
        elif kind == "summary":
            ate[r["track"]] = r["text"]
        elif kind == "poses":
            full[r["track"]] = (np.array(r["frames"]), np.array(r["poses"], float))
        elif kind == "closures":
            closures[r["track"]] = (np.array(r["pairs"], int).reshape(-1, 2),
                                    None if r.get("T_ji") is None else np.array(r["T_ji"], float))
        elif kind == "reference" and not len(reference):
            reference = np.array(r["positions"], float)
        elif kind == "note":
            m = re.fullmatch(r"ATE ([\d.]+) m -> ([\d.]+) m", r.get("text", ""))
            noted = m.groups() if m else noted

    fronts = [t for t in last if not hints.get(t, {}).get("derived_from")]
    predictions = []
    for track, positions in last.items():
        parent = hints.get(track, {}).get("derived_from")
        name = (track[len(parent):].strip() if parent and track.startswith(parent)
                else "odometry" if len(fronts) == 1
                else track.replace(f" - {scene}", ""))
        text = ate.get(track)
        if text is None and noted and len(last) == 2:
            text = f"ATE {noted[1 if parent else 0]} m"
        predictions.append(Prediction(path, track, name, bool(parent),
                                      np.array(positions, float), text,
                                      *full.get(track, (None, None)),
                                      *closures.get(track, closures.get(parent, (None, None)))))
    stamp = path.name[:15]    # RunWriter names files YYYYmmdd-HHMMSS_...
    return RunEntry(
        log=path, when=f"{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}",
        config=fronts[0].replace(f" - {scene}", "") if len(fronts) == 1 else f"{len(fronts)} variants",
        predictions=predictions, reference=reference)
