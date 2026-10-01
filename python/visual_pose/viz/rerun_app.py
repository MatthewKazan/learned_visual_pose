"""
The viewer: keeps a rerun viewer in step with the data (data_utils.sources)
and run.py's logs (out/runs). One recording per scene; runs appear as they
finish, a run still being written is drawn live, and a scene's frames and maps
are sent the first time its recording is opened.

    .venv/bin/python -m visual_pose.viz.rerun_app           # leave open
    .venv/bin/python -m visual_pose.viz.rerun_app --follow  # jump to each scene a run finishes in

A new data type needs only a Source; per-type extra layers go in EXTRAS.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from visual_pose.data_utils.sources import SOURCES, HM3DSource, Source, source
from visual_pose.viz import rerun_view as view
from visual_pose.viz.run_log import DEFAULT_ROOT as RUNS_ROOT, SUFFIX, read_run_log

PORT = 9876
VIEWER_MEMORY = "4GB"
SCAN_S, TAIL_S, ACTIVE_S = 2.0, 0.25, 1.0
WORK_BUDGET_S = 1.0       # queued work per loop pass, so tails and clicks stay responsive
RUNS_PER_SCENE = 4        # newest finished runs loaded per scene at startup
LIVE_WINDOW_S = 600       # an unclosed log modified this recently is live; older ones died
# Source type -> (rec, src, renderer) -> info lines. The renderer is a SceneRenderer
# for the source's scene, opened only for these.
EXTRAS = {HM3DSource: view.log_hm3d_extras}


def first_record(path: Path) -> dict:
    try:
        with path.open() as f:
            return json.loads(f.readline())
    except (OSError, json.JSONDecodeError):
        return {}


def is_closed(path: Path) -> bool:
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 200))
        return b'"kind":"closed"' in f.read()


def reference_of(path: Path) -> np.ndarray:
    with path.open() as f:
        for line in f:
            if '"kind":"reference"' in line:
                return np.array(json.loads(line)["positions"], float)
    return np.zeros((0, 3))


class Tail:
    """Complete lines appended to a log since the last poll."""

    def __init__(self, path: Path):
        self.handle, self.pending = path.open(), ""

    def poll(self) -> list[dict]:
        out = []
        while chunk := self.handle.readline():
            if not chunk.endswith("\n"):
                self.pending += chunk
                break
            line, self.pending = (self.pending + chunk).strip(), ""
            if line:
                out.append(json.loads(line))
        return out


@dataclass
class Scene:
    key: str
    rec: object
    src: Source
    seq: object                                 # src.frames()
    stamp: float = 0.0
    runs: dict = field(default_factory=dict)    # log -> RunEntry
    colors: dict = field(default_factory=dict)  # (log, track) -> hex
    live: dict = field(default_factory=dict)    # log -> Tail
    extra_lines: list = field(default_factory=list)
    heavy: bool = False

    @property
    def T_true(self):
        return self.seq.poses if self.seq.has_ground_truth else None


class App:
    def __init__(self, runs_root: Path, *, port: int = PORT, headless: bool = False,
                 sources: list = SOURCES, follow: bool = False):
        import rerun as rr
        from rerun.experimental import ViewerClient
        self.rr, self.runs_root, self.sources, self.follow = rr, runs_root, sources, follow
        self.starting = True      # the viewer jumps to every new recording at startup anyway
        # spawn only if nothing listens: a viewer too busy to answer once got a second one beside it
        with socket.socket() as probe:
            running = probe.connect_ex(("127.0.0.1", port)) == 0
        if running:
            self.client = ViewerClient.connect(f"rerun+http://127.0.0.1:{port}/proxy")
            self.client.close_recordings("all")        # a restart re-sends everything
        else:
            self.client = ViewerClient.spawn(port=port, memory_limit=VIEWER_MEMORY, headless=headless,
                                             hide_welcome_screen=True, executable_path=view.viewer_executable())
        self.scenes: dict[str, Scene] = {}
        self.seen_logs: set[Path] = set()
        self.unreadable: dict[str, float] = {}         # key -> stamp: retried only when the data changes
        self.work: deque = deque()
        self.renderer = None
        self.tables = self._stream("ATE tables", "ATE tables", frames=False)
        self._last = dict.fromkeys(("scan", "tail", "active"), 0.0)

    # -- recordings

    def _stream(self, key: str, name: str, *, frames: bool, hidden=()):
        # one application id per scene: rerun keeps one active blueprint per app id
        rec = self.rr.RecordingStream(name, recording_id=key)
        rec.connect_grpc(self.client.url)
        self.rr.send_recording_name(name, recording=rec)
        # active, not only default: the viewer keeps a layout per app id across restarts,
        # and an old one (camera panel on world/truth/camera) left the camera blank
        self.rr.send_blueprint(view.blueprint(list(hidden), frames=frames), recording=rec)
        return rec

    def _renderer(self, scene_dir: Path):
        if self.renderer is None or self.renderer.scene_dir != scene_dir:
            self._release_renderer()
            from visual_pose.data_utils.hm3d_renderer import SceneRenderer
            self.renderer = SceneRenderer(scene_dir)
        return self.renderer

    def _release_renderer(self) -> None:
        # not held while idle: run.py renders on the same GPU
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _scene(self, key: str) -> Scene | None:
        """None when the key's data cannot be read: missing (deleted, another machine) or empty."""
        src = source(key)
        try:
            seq, stamp = src.frames(), src.stamp()
            if not len(seq):
                raise ValueError("no frames")
        except (OSError, StopIteration, ValueError):
            return None
        name = key.removeprefix("hm3d:").replace("/", " / ")
        scene = Scene(key, self._stream(key, name, frames=True, hidden=view.SCENE_HIDDEN), src, seq, stamp=stamp)
        view.log_scene(scene.rec, src.axes, scene.T_true[:, :3, 3] if scene.T_true is not None else np.zeros((0, 3)))
        if type(src) in EXTRAS:
            scene.extra_lines = EXTRAS[type(src)](scene.rec, src, self._renderer(src.dir.parents[1]))
        self._info(scene)
        return scene

    def _info(self, scene: Scene) -> None:
        lines = [f"{len(scene.seq)} frames" + ("" if scene.T_true is not None else " · no ground truth")]
        lines += scene.extra_lines
        if scene.live:
            lines.append(f"**live:** {', '.join(p.name[:15] for p in scene.live)}")
        view.log_info(scene.rec, scene.key, lines, scene.runs.values())

    def _add_run(self, scene: Scene, log: Path) -> None:
        entry = scene.runs[log] = read_run_log(log)
        for p in entry.predictions:
            scene.colors.setdefault((log, p.track), view.PRED_COLORS[len(scene.colors) % len(view.PRED_COLORS)])
        colors = {p.track: scene.colors[log, p.track] for p in entry.predictions}
        placed = {p.track: view.place(scene.T_true, p) for p in entry.predictions}
        view.log_run(scene.rec, scene.seq, scene.T_true, entry, placed, colors)
        if scene.heavy:
            self._clouds(scene, entry, placed, colors)
        # make_active switches the viewer to this recording; without it the new
        # default applies on the recording's "reset blueprint"
        self.rr.send_blueprint(view.blueprint(view.hidden_by_default(scene.runs.values(), scene.T_true is not None),
                                              frames=True), recording=scene.rec,
                               make_active=self.follow or self.starting)
        self._info(scene)

    def _clouds(self, scene: Scene, entry, placed, colors) -> None:
        view.log_run_clouds(scene.rec, view.samples(scene.src, scene.seq), scene.seq, scene.src.max_depth,
                            entry, placed, colors, own_colors=scene.T_true is None)

    def _open(self, scene: Scene) -> None:
        print(f"opening {scene.key}: frames and maps")
        view.log_frames(scene.rec, view.samples(scene.src, scene.seq), scene.seq, scene.T_true,
                        scene.src.max_depth)
        for log, entry in scene.runs.items():
            colors = {p.track: scene.colors[log, p.track] for p in entry.predictions}
            self._clouds(scene, entry, {p.track: view.place(scene.T_true, p) for p in entry.predictions}, colors)
        scene.heavy = True

    # -- discovery

    def scan(self) -> None:
        for cls in self.sources:
            for key in cls.discover():
                stamp = source(key).stamp()
                if self.unreadable.get(key) != stamp and (key not in self.scenes or self.scenes[key].stamp != stamp):
                    self.work.append(("scene", key))
        per_scene: dict[str, int] = {}
        for log in sorted(self.runs_root.glob(f"*{SUFFIX}"), reverse=True):     # newest first
            if log in self.seen_logs or not (key := first_record(log).get("name")):
                continue
            self.seen_logs.add(log)
            closed = is_closed(log)
            live = not closed and time.time() - log.stat().st_mtime < LIVE_WINDOW_S
            if not (closed or live):
                continue                                  # died without closing
            if closed and key not in self.scenes:
                per_scene[key] = per_scene.get(key, 0) + 1
                if per_scene[key] > RUNS_PER_SCENE:
                    continue
            self.work.append(("log", log, key, live))

    def _do(self, item) -> None:
        if item[0] == "scene":
            key = item[1]
            if key in self.scenes and self.scenes[key].stamp == source(key).stamp():
                return                                    # queued twice before it was done
            if old := self.scenes.pop(key, None):        # the data changed: its runs are rechecked
                self.client.close_recordings([r.store_id for r in self.client.viewer_state().recordings
                                              if r.store_id.endswith(":" + key)])
                self.seen_logs -= set(old.runs) | set(old.live)
            if scene := self._scene(key):
                self.scenes[key] = scene
            else:
                self.unreadable[key] = source(key).stamp()
            return
        _, log, key, live = item
        scene = self.scenes.get(key) or self._scene(key)
        if scene is None:
            return                                        # its data is gone: nothing to draw the run on
        self.scenes[key] = scene
        if scene.T_true is not None and view.is_stale(scene.T_true, reference_of(log)):
            return
        if live:
            scene.live[log] = Tail(log)
            self._info(scene)
        else:
            self._add_run(scene, log)

    # -- live logs

    def tail(self) -> None:
        for scene in list(self.scenes.values()):
            for log, t in list(scene.live.items()):
                for r in t.poll():
                    self._live_record(scene, log, r)

    def _live_record(self, scene: Scene, log: Path, r: dict) -> None:
        rr, kind = self.rr, r.get("kind")
        if kind == "frame":
            P = np.array(r["positions"], float)
            if scene.T_true is not None:
                P = P @ scene.T_true[0, :3, :3].T + scene.T_true[0, :3, 3]
            color = view._hex(view.PRED_COLORS[zlib.crc32(r["track"].encode()) % len(view.PRED_COLORS)])
            path = f"world/live/{view.slug(r['track'])[-40:]}"
            scene.rec.log(path, rr.LineStrips3D([P], colors=[color], radii=0.03), static=True)
            scene.rec.log(f"{path}/head", rr.Points3D(P[-1:], colors=[color], radii=0.12), static=True)
        elif kind == "table":
            self._table(log, r)
        elif kind == "closed":
            del scene.live[log]
            scene.rec.log("world/live", rr.Clear(recursive=True), static=True)
            self._add_run(scene, log)

    def _table(self, log: Path, r: dict) -> None:
        self.tables.log("table", self.rr.TextDocument(
            f"**{r['title']}** · written {log.name[:15]}\n\n```\n{r['text']}\n```",
            media_type=self.rr.MediaType.MARKDOWN), static=True)

    def latest_table(self) -> None:
        for log in sorted(self.runs_root.glob(f"*{SUFFIX}"), reverse=True)[:40]:
            with log.open("rb") as f:
                f.seek(max(0, log.stat().st_size - 20000))
                for line in f.read().decode(errors="ignore").splitlines():
                    if '"kind":"table"' in line:
                        return self._table(log, json.loads(line))

    # -- the loop

    def check_active(self) -> None:
        try:
            active = str(self.client.viewer_state().active_recording)
        except TimeoutError:                           # viewer busy (seen opening a scene under memory pressure)
            return
        # the store id ends ":<recording id>"; the app id before it is rewritten by the viewer
        scene = next((s for k, s in self.scenes.items() if active.endswith(":" + k)), None)
        if scene is not None and not scene.heavy:
            self._open(scene)

    def step(self) -> None:
        now = time.time()
        if now - self._last["scan"] > SCAN_S:
            self._last["scan"] = now
            self.scan()
        had_work, deadline = bool(self.work), time.time() + WORK_BUDGET_S
        while self.work and time.time() < deadline:
            self._do(self.work.popleft())
        if had_work and not self.work:
            self._release_renderer()
            self.starting = False
        if now - self._last["tail"] > TAIL_S:
            self._last["tail"] = now
            self.tail()
        # not during startup: the viewer switches to each recording as it arrives
        if not self.work and now - self._last["active"] > ACTIVE_S:
            self._last["active"] = now
            self.check_active()

    def run(self) -> None:
        self.latest_table()
        print(f"watching {[c.__name__ for c in self.sources]} and {self.runs_root}; Ctrl-C to stop")
        while True:
            self.step()
            time.sleep(0.05)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RUNS_ROOT)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--follow", action="store_true", help="switch to each scene a run finishes in")
    args = ap.parse_args()
    App(args.runs, port=args.port, follow=args.follow).run()


if __name__ == "__main__":
    main()
