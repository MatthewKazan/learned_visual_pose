"""
Live trajectory viewer -- a generic animation harness for iterative solvers.

Solver and plot are decoupled. A *producer* is any callable taking a
TrackEmitter and calling `send()` once per step; it runs on its own thread and
never touches matplotlib. The app owns the figure, drains whatever arrived since
the last frame and redraws at a fixed rate -- so a producer emitting 500 steps a
second is coalesced into `fps` redraws, not 500.

The window is a long-lived session: `start()` once, then `submit()` as many
producers as you like, whenever you like, with whatever parameters. Adding an
optimiser (a learned pose-graph model, another backend) means writing a
producer. Nothing in this file changes.

Threading contract:
  * producers only ever touch their TrackEmitter, which is a queue put
  * every artist is created, mutated and removed inside `_draw`, i.e. on the
    thread running the GUI event loop. matplotlib is not thread-safe and
    creating a line from a worker thread corrupts the figure in ways that
    surface much later.
  * a producer inside a pybind11 call holds the GIL for that call, so one C++
    step taking 200 ms freezes the redraw for 200 ms. Fine per-step; not fine
    for a whole solve behind one call.
"""
from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# Which two axes a 2D view shows. "3d" is handled separately.
PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

DEFAULT_COLORS = ("#2563eb", "#f97316", "#7c3aed", "#059669", "#dc2626", "#0891b2",
                  "#c026d3", "#65a30d", "#e11d48", "#0d9488", "#a16207", "#4f46e5")

# Two at most: a wide legend eats the horizontal space the trajectory needs, and
# long labels are unreadable once they wrap.
LEGEND_MAX_COLUMNS = 2

# Dark and desaturated, so a truth line reads as a reference rather than as one
# more estimate -- but distinct, because two sequences means two truths.
REFERENCE_COLORS = ("#111827", "#6b7280", "#7f1d1d", "#1e3a8a", "#365314")

# Scalar series beyond this many distinct keys share the last panel rather than
# shrinking every panel into a sliver.
MAX_PANELS = 4


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


@dataclass(frozen=True)
class Snapshot:
    """One step of one track, as handed across the thread boundary."""
    track: str
    positions: np.ndarray             # (N, 3)
    step: int                         # this track's own step counter
    scalars: Mapping[str, float] = field(default_factory=dict)
    x: float | None = None            # x for the scalars; defaults to `step`
    x_label: str = "iteration"
    edges: np.ndarray | None = None   # (M, 2) vertex index pairs, optional
    keep: bool = False                # retain as a ghost regardless of ghost_every
    final: bool = False               # last word from this track


class TrackEmitter:
    """A producer's handle on one track. Owned by that producer's thread."""

    def __init__(self, app: LiveTrajectoryApp, track: str):
        self.app = app
        self.track = track
        self._count = 0
        self._last: np.ndarray | None = None

    def send(self, poses, *, x: float | None = None, x_label: str = "iteration",
             edges=None, keep: bool = False, final: bool = False, **scalars) -> bool:
        """Push one frame. Returns False once the track is done.
    """
        self.app.pause_gate.wait()
        if self.app.closed or not self.app.knows(self.track):
            return False
        P = positions_of(poses)
        self._last = P
        self.app.queue.put(Snapshot(
            track=self.track, positions=P, step=self._count,
            # A NaN reaches relim() and makes the whole panel's limits NaN, which
            # blanks every series on it. A failed step has no scalar, not a bad one.
            scalars={k: float(v) for k, v in scalars.items()
                     if v is not None and np.isfinite(v)},
            x=x, x_label=x_label,
            edges=None if edges is None else np.asarray(edges, dtype=int),
            keep=keep, final=final,
        ))
        self._count += 1
        return True

    def finish(self) -> None:
        """Mark the track complete, freezing its last estimate."""
        if self._last is not None:
            self.app.queue.put(Snapshot(self.track, self._last,
                                        max(self._count - 1, 0), final=True))


@dataclass
class _Track:
    color: str
    ghost_every: int | None
    show_edges: bool
    # Dotted by default: solid is reserved for the reference, so "is this the
    # truth or an estimate" is answerable without reading the legend.
    linestyle: str = ":"
    live: object = None                 # Line2D / Line3D, overwritten in place
    edges: object = None                # LineCollection, overwritten in place
    ghosts: list = field(default_factory=list)
    last: np.ndarray | None = None      # so a rebuilt figure can redraw it
    history: dict = field(default_factory=dict)        # scalar key -> (xs, ys)
    history_labels: dict = field(default_factory=dict)  # scalar key -> x axis label
    scalar_lines: dict = field(default_factory=dict)
    step: int = 0
    done: bool = False


class LiveTrajectoryApp:
    """
    Left: trajectories, one live line per track plus faded ghosts of earlier
    steps. Right: one panel per scalar key the producers reported.

    Keys: space pauses, g toggles ghosts, c clears everything on screen,
    q closes the window. Click a legend entry to hide that line.
    """

    def __init__(self, *, plane: str = "xy", fps: int = 20, title: str = "live solver",
                 ghost_every: int | None = None, max_ghosts: int = 6,
                 scalar_scale: str = "log", figsize=(14, 7)):
        if plane != "3d" and plane not in PLANES:
            raise ValueError(f"plane must be '3d' or one of {sorted(PLANES)}")
        self.plane = plane
        self.fps = fps
        self.title = title
        self.ghost_every = ghost_every
        self.max_ghosts = max_ghosts
        self.scalar_scale = scalar_scale
        self.figsize = figsize

        self.queue: queue.Queue[Snapshot] = queue.Queue()
        self.pause_gate = threading.Event()
        self.pause_gate.set()
        self.closed = False

        self._lock = threading.Lock()
        self._tracks: dict[str, _Track] = {}
        self._pending_removals: list[str] = []
        # Several at once: comparing two sequences means two ground truths.
        self._references: dict[str, np.ndarray] = {}
        self._reference_lines: dict[str, object] = {}
        self._references_dirty = False
        self._bounds: np.ndarray | None = None   # (2, 3) min/max over everything
        self._auto_color = 0
        self._show_ghosts = True
        self._hidden: set[tuple[str, str]] = set()
        self._threads: list[threading.Thread] = []
        self._panels: dict[str, object] = {}     # scalar key -> Axes
        self._legend_map: dict = {}
        self._fig = None
        self._anim = None

    # -- session -------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._fig is not None and not self.closed

    def start(self, *, interactive: bool = True) -> LiveTrajectoryApp:
        """Open the window and block until it closes.
    """
        if self.started:
            return self
        self.closed = False
        self._reset_artists()
        self._build_figure()
        self._start_animation()
        if interactive:
            plt.ion()
            plt.show(block=False)
        return self

    def submit(self, name: str, producer: Callable[[TrackEmitter], None], *,
               color: str | None = None, ghost_every: int | None = None,
               show_edges: bool = False, linestyle: str = ":",
               replace_existing: bool = False) -> str:
        """Queue work for the matplotlib thread.
    """
        if replace_existing and name in self._tracks:
            self.remove(name)
        track_name = self.add_track(name, color=color, ghost_every=ghost_every,
                                    show_edges=show_edges, linestyle=linestyle)
        if not self.started:
            self.start()
        else:
            self._start_animation()      # it stops itself when everything is done
        thread = threading.Thread(target=self._guarded,
                                  args=(producer, TrackEmitter(self, track_name)),
                                  name=f"producer:{track_name}", daemon=True)
        thread.start()
        self._threads.append(thread)
        return track_name

    def spawn(self, worker: Callable[[LiveTrajectoryApp], None], *,
              name: str = "worker") -> None:
        """Run `fn` on a daemon thread.
    """
        if not self.started:
            self.start()
        else:
            self._start_animation()
        thread = threading.Thread(target=self._guarded_worker, args=(worker,),
                                  name=f"worker:{name}", daemon=True)
        thread.start()
        self._threads.append(thread)

    def _guarded_worker(self, worker: Callable[[LiveTrajectoryApp], None]) -> None:
        try:
            worker(self)
        except Exception as exc:   # noqa: BLE001 - reported, not handled
            print(f"[worker] failed: {exc!r}")
            import traceback
            traceback.print_exc()

    def run(self, producers: Mapping[str, Callable[[TrackEmitter], None]]) -> None:
        """Submit several producers and block until the window closes."""
        plt.ioff()                       # or show() returns instantly
        self.start(interactive=False)
        for name, fn in producers.items():
            self.submit(name, fn)
        plt.show()
        self.closed = True
        self.pause_gate.set()

    def remove(self, name: str) -> None:
        """Drop a track and its artists. Its producer stops at the next send()."""
        with self._lock:
            if name in self._tracks:
                self._pending_removals.append(name)
        self._start_animation()          # the removal happens inside _draw

    def clear(self, *, reference: bool = False) -> None:
        """Drop every track, so a long session does not accumulate twenty lines."""
        with self._lock:
            self._pending_removals.extend(self._tracks)
        if reference:
            self._references = {}
            self._references_dirty = True
        self._start_animation()

    def knows(self, name: str) -> bool:
        """True if this track has been seen.
    """
        with self._lock:
            return name in self._tracks and name not in self._pending_removals

    def tracks(self) -> list[str]:
        with self._lock:
            return [n for n in self._tracks if n not in self._pending_removals]

    # -- setup ---------------------------------------------------------------

    def add_reference(self, poses, label: str = "ground truth") -> None:
        """A solid line the estimates are compared against. Keyed by label.
    """
        self._references[label] = positions_of(poses)
        self._references_dirty = True

    def add_track(self, name: str, *, color: str | None = None,
                  ghost_every: int | None = None, show_edges: bool = False,
                  linestyle: str = ":") -> str:
        """Register a track, returning the unique name it was given."""
        name = self._unique(name)
        if color is None:
            color = DEFAULT_COLORS[self._auto_color % len(DEFAULT_COLORS)]
            self._auto_color += 1
        self._tracks[name] = _Track(
            color=color,
            ghost_every=self.ghost_every if ghost_every is None else ghost_every,
            show_edges=show_edges,
            linestyle=linestyle,
        )
        return name

    def color_of(self, name: str) -> str | None:
        """So a producer can give a derived track the same colour as its parent."""
        track = self._tracks.get(name)
        return None if track is None else track.color

    def emitter(self, name: str, **kwargs) -> TrackEmitter:
        """Handle for pushing frames to one named track.
    """
        track_name = self.add_track(name, **kwargs)
        self._start_animation()
        return TrackEmitter(self, track_name)

    def _unique(self, name: str) -> str:
        if name not in self._tracks:
            return name
        for n in itertools.count(2):
            candidate = f"{name} #{n}"
            if candidate not in self._tracks:
                return candidate
        raise AssertionError("unreachable")

    def run_headless(self, producers: Mapping[str, Callable[[TrackEmitter], None]],
                     timeout: float = 60.0) -> dict[str, Snapshot]:
        """
        Drive producers with no figure at all and return each track's last
        snapshot. Used to test producers without a display.
        """
        names = [self.add_track(n) for n in producers]
        emitters = dict(zip(names, (TrackEmitter(self, n) for n in names)))
        threads = [threading.Thread(target=self._guarded, args=(fn, emitters[n]),
                                    daemon=True)
                   for n, fn in zip(names, producers.values())]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout)
        last: dict[str, Snapshot] = {}
        while True:
            try:
                snap = self.queue.get_nowait()
            except queue.Empty:
                break
            if not snap.scalars and snap.track in last:
                snap = replace(snap, scalars=last[snap.track].scalars)
            last[snap.track] = snap
        return last

    @staticmethod
    def _guarded(fn: Callable[[TrackEmitter], None], emitter: TrackEmitter) -> None:
        # A producer raising in its own thread would otherwise vanish into
        # threading's default handler and look like a track that simply stalled.
        try:
            fn(emitter)
        except Exception as exc:   # noqa: BLE001 - reported, not handled
            print(f"[{emitter.track}] producer failed: {exc!r}")
            import traceback
            traceback.print_exc()

    # -- figure --------------------------------------------------------------

    def _start_animation(self) -> None:
        if self._anim is None:
            # cache_frame_data=False: frames is unbounded, and caching it would
            # grow without limit for a window left open for minutes.
            self._anim = FuncAnimation(self._fig, self._draw, frames=itertools.count(),
                                       interval=1000 // self.fps, blit=False,
                                       cache_frame_data=False)
        elif self._fig is not None:
            self._anim.event_source.start()

    def _reset_artists(self) -> None:
        """Forget every artist, keeping track state, so the figure can be rebuilt."""
        for track in self._tracks.values():
            track.live = None
            track.edges = None
            track.ghosts = []
            track.scalar_lines = {}
        self._panels = {}
        self._reference_lines = {}
        self._references_dirty = bool(self._references)
        self._bounds = None
        self._anim = None

    def _build_figure(self) -> None:
        self._fig = plt.figure(figsize=self.figsize)
        # Kept as a gridspec cell so scalar panels can be added later: the right
        # cell is subdivided on demand, see _panel_for.
        self._outer = self._fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
        # The legend lives in a strip of its own beneath the plot rather than
        # inside it: a dozen lines (three estimators on two sequences, each with
        # a graph twin) covered half the trajectory.
        left = self._outer[0, 0].subgridspec(2, 1, height_ratios=[1, 0.22],
                                             hspace=0.05)
        if self.plane == "3d":
            self.ax = self._fig.add_subplot(left[0], projection="3d")
            self.ax.set_zlabel("z")
        else:
            self.ax = self._fig.add_subplot(left[0])
            self.ax.set_aspect("equal", adjustable="box")
        self._legend_ax = self._fig.add_subplot(left[1])
        self._legend_ax.axis("off")
        i, j = (0, 1) if self.plane == "3d" else PLANES[self.plane]
        self.ax.set_xlabel("xyz"[i])
        self.ax.set_ylabel("xyz"[j])
        self.ax.grid(alpha=0.25)

        annotate = self.ax.text2D if self.plane == "3d" else self.ax.text
        self._status = annotate(
            0.01, 0.99, "", transform=self.ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace", color="#374151", zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
        self._fig.canvas.mpl_connect("close_event", self._on_close)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig.canvas.mpl_connect("pick_event", self._on_pick)
        self._reconcile()

    def _line(self, ax, **kw):
        if self.plane == "3d":
            (line,) = ax.plot([], [], [], **kw)
        else:
            (line,) = ax.plot([], [], **kw)
        return line

    def _set(self, line, P: np.ndarray) -> None:
        if self.plane == "3d":
            line.set_data_3d(P[:, 0], P[:, 1], P[:, 2])
        else:
            i, j = PLANES[self.plane]
            line.set_data(P[:, i], P[:, j])

    def _update_edges(self, track: _Track, P: np.ndarray, pairs: np.ndarray) -> None:
        """Redraw the edge overlay for one track.
    """
        if not len(pairs):
            return
        segments = P[pairs]                          # (M, 2, 3)
        if self.plane != "3d":
            i, j = PLANES[self.plane]
            segments = segments[:, :, [i, j]]
        segments = list(segments)

        if track.edges is None:
            if self.plane == "3d":
                from mpl_toolkits.mplot3d.art3d import Line3DCollection
                track.edges = Line3DCollection(segments, colors=track.color,
                                               linewidths=0.6, alpha=0.35)
                self.ax.add_collection3d(track.edges)
            else:
                track.edges = LineCollection(segments, colors=track.color,
                                             linewidths=0.6, alpha=0.35)
                self.ax.add_collection(track.edges)
        else:
            track.edges.set_segments(segments)

    def _panel_for(self, key: str, x_label: str):
        """Axes for one scalar series, created on demand.
    """
        if key in self._panels:
            return self._panels[key]
        if len(self._panels) >= MAX_PANELS:
            return next(reversed(self._panels.values()))

        rows = len(self._panels) + 1
        sub = self._outer[0, 1].subgridspec(rows, 1, hspace=0.55)
        for row, ax in enumerate(self._panels.values()):
            ax.set_subplotspec(sub[row])
        ax = self._fig.add_subplot(sub[rows - 1])
        ax.set_yscale(self.scalar_scale)
        ax.set_ylabel(key, fontsize=8)
        ax.set_xlabel(x_label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        self._panels[key] = ax
        return ax

    def _legend(self) -> None:
        # References first, then tracks -- matching artist creation order.
        names = ([("ref", label) for label in self._reference_lines]
                 + [("track", name) for name in self._tracks])
        handles = (list(self._reference_lines.values())
                   + [t.live for t in self._tracks.values()])
        self._legend_map = {}
        existing = self._legend_ax.get_legend()
        if not names:
            if existing is not None:
                existing.remove()
            return
        # Measure, do not assume: the strip is ~39% of the figure and two
        # columns of long labels render at ~55%, drawing over the panels. Try
        # widest-first and keep the first that fits.
        wanted = min(LEGEND_MAX_COLUMNS, 1 + (len(names) - 1) // 5)
        texts = [str(name) for _, name in names]
        available = self._legend_ax.get_window_extent().transformed(
            self._fig.transFigure.inverted()).width
        legend = None
        for columns in range(wanted, 0, -1):
            for fontsize in (7 if len(names) <= 10 else 6, 6, 5, 4):
                # loc anchors the legend's TOP to the strip, so an overflow
                # grows down off the figure rather than up over the trajectory.
                legend = self._legend_ax.legend(
                    handles, texts, loc="upper center", ncols=columns,
                    fontsize=fontsize, frameon=False, handlelength=1.6,
                    columnspacing=1.2, borderaxespad=0.0)
                self._fig.canvas.draw_idle()
                width = legend.get_window_extent().transformed(
                    self._fig.transFigure.inverted()).width
                if width <= available:
                    break
            else:
                continue
            break
        for entry, name in zip(legend.get_lines(), names):
            entry.set_picker(6)
            self._legend_map[entry] = name
        self._refresh_visibility()

    # -- interaction ---------------------------------------------------------

    def _on_close(self, _event) -> None:
        self.closed = True
        self.pause_gate.set()           # release any producer blocked on pause

    def _on_key(self, event) -> None:
        if event.key == " ":
            if self.pause_gate.is_set():
                self.pause_gate.clear()
            else:
                self.pause_gate.set()
        elif event.key == "g":
            self._show_ghosts = not self._show_ghosts
            self._refresh_visibility()
        elif event.key == "c":
            self.clear(reference=True)
        elif event.key == "q":
            plt.close(self._fig)

    def _on_pick(self, event) -> None:
        if event.artist not in self._legend_map:
            return
        self._hidden.symmetric_difference_update({self._legend_map[event.artist]})
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        """Single source of truth: hidden tracks and the ghost toggle."""
        for label, line in self._reference_lines.items():
            line.set_visible(("ref", label) not in self._hidden)
        for name, track in self._tracks.items():
            shown = ("track", name) not in self._hidden
            if track.live is not None:
                track.live.set_visible(shown)
            if track.edges is not None:
                track.edges.set_visible(shown)
            for ghost in track.ghosts:
                ghost.set_visible(shown and self._show_ghosts)
            for line in track.scalar_lines.values():
                line.set_visible(shown)
        for entry, key in self._legend_map.items():
            entry.set_alpha(0.25 if key in self._hidden else 1.0)

    # -- per-frame -----------------------------------------------------------

    def _reconcile(self) -> bool:
        """
        Bring artists in line with the track dict. Runs on the GUI thread only.
        Returns True if anything was created or destroyed.
        """
        changed = False

        with self._lock:
            removals, self._pending_removals = self._pending_removals, []
        for name in removals:
            track = self._tracks.pop(name, None)
            if track is None:
                continue
            for artist in [track.live, track.edges, *track.ghosts,
                           *track.scalar_lines.values()]:
                if artist is not None:
                    artist.remove()
            self._hidden.discard(("track", name))
            changed = True

        if self._references_dirty:
            for label in list(self._reference_lines):
                self._reference_lines.pop(label).remove()
            for order, (label, positions) in enumerate(self._references.items()):
                line = self._line(self.ax, ls="-", lw=1.8, zorder=1, alpha=0.9,
                                  color=REFERENCE_COLORS[order % len(REFERENCE_COLORS)],
                                  label=label)
                self._set(line, positions)
                self._reference_lines[label] = line
            self._references_dirty = False
            changed = True

        for name, track in self._tracks.items():
            if track.live is None:
                track.live = self._line(self.ax, color=track.color, lw=2.0,
                                        ls=track.linestyle, zorder=3, label=name)
                if track.last is not None:
                    self._set(track.live, track.last)
                changed = True

        if changed:
            # Removing a track can shrink the view, so recompute from scratch
            # rather than only ever growing.
            self._bounds = None
            for P in (list(self._references.values())
                      + [t.last for t in self._tracks.values() if t.last is not None]):
                self._grow(P)
            self._apply_bounds()
            self._legend()
        return changed

    def _draw(self, _frame):
        rebuilt = self._reconcile()
        latest: dict[str, Snapshot] = {}
        kept: list[Snapshot] = []
        grew = False

        # Drain everything the producers queued since the last frame. Only the
        # newest snapshot per track is drawn live; the ones a ghost policy asks
        # for are drawn once and left behind.
        while True:
            try:
                snap = self.queue.get_nowait()
            except queue.Empty:
                break
            track = self._tracks.get(snap.track)
            if track is None:                  # removed while its producer ran
                continue
            latest[snap.track] = snap
            track.step = snap.step
            track.last = snap.positions
            if snap.keep or (track.ghost_every and snap.step % track.ghost_every == 0):
                kept.append(snap)
            if snap.final:
                track.done = True
            for key, value in snap.scalars.items():
                xs, ys = track.history.setdefault(key, ([], []))
                xs.append(snap.step if snap.x is None else snap.x)
                ys.append(value)
                track.history_labels[key] = snap.x_label

        for snap in kept:
            self._add_ghost(snap)

        for name, snap in latest.items():
            track = self._tracks[name]
            self._set(track.live, snap.positions)
            track.live.set_linewidth(2.6 if track.done else 2.0)
            if track.show_edges and snap.edges is not None:
                self._update_edges(track, snap.positions, snap.edges)
            grew |= self._grow(snap.positions)

        if latest:
            self._draw_scalars()
        if grew:
            self._apply_bounds()
        self._retitle()

        idle = (self.queue.empty() and not rebuilt
                and all(t.done for t in self._tracks.values()))
        if idle and self._anim is not None:
            # Nothing left to redraw. submit()/remove() restart the event source.
            self._anim.event_source.stop()
        return ()

    def _add_ghost(self, snap: Snapshot) -> None:
        track = self._tracks[snap.track]
        ghost = self._line(self.ax, color=track.color, lw=1.0, alpha=0.28,
                           ls=track.linestyle, zorder=2)
        self._set(ghost, snap.positions)
        ghost.set_visible(self._show_ghosts
                          and ("track", snap.track) not in self._hidden)
        track.ghosts.append(ghost)
        # Bounded so a long solve cannot accumulate hundreds of artists and turn
        # each redraw into a slideshow.
        while len(track.ghosts) > self.max_ghosts:
            track.ghosts.pop(0).remove()

    def _draw_scalars(self) -> None:
        touched = set()
        for name, track in self._tracks.items():
            for key, (xs, ys) in track.history.items():
                line = track.scalar_lines.get(key)
                if line is None:
                    ax = self._panel_for(key, track.history_labels.get(key, "iteration"))
                    (line,) = ax.plot([], [], color=track.color, lw=1.5, ms=3,
                                      ls=track.linestyle,
                                      marker="." if len(xs) < 200 else None,
                                      label=name)
                    track.scalar_lines[key] = line
                line.set_data(xs, ys)
                touched.add(line.axes)
        for ax in touched:
            ax.relim()
            ax.autoscale_view()

    def _grow(self, P: np.ndarray) -> bool:
        lo, hi = P.min(axis=0), P.max(axis=0)
        if self._bounds is None:
            self._bounds = np.stack([lo, hi])
            return True
        new = np.stack([np.minimum(self._bounds[0], lo), np.maximum(self._bounds[1], hi)])
        if np.array_equal(new, self._bounds):
            return False
        self._bounds = new
        return True

    def _apply_bounds(self) -> None:
        if self._bounds is None:
            return
        lo, hi = self._bounds
        pad = np.maximum((hi - lo) * 0.08, 0.1)
        lo, hi = lo - pad, hi + pad
        if self.plane == "3d":
            self.ax.set_xlim(lo[0], hi[0])
            self.ax.set_ylim(lo[1], hi[1])
            self.ax.set_zlim(lo[2], hi[2])
            # Equal axes, or the shape on screen is not the data's shape.
            # set_aspect("equal") is 2D-only, so 3D needs the box proportioned
            # to the data and re-proportioned as the bounds grow.
            self.ax.set_box_aspect(np.maximum(hi - lo, 1e-6))
        else:
            i, j = PLANES[self.plane]
            self.ax.set_xlim(lo[i], hi[i])
            self.ax.set_ylim(lo[j], hi[j])

    def _retitle(self) -> None:
        running = sum(not t.done for t in self._tracks.values())
        state = "paused" if not self.pause_gate.is_set() else (
            f"{running} running" if running else "idle")
        self.ax.set_title(f"{self.title}   [{state}]", fontsize=10)
        # Only the running tracks: finished ones are named in the legend, and
        # this box would otherwise cover the plot after a few submissions.
        active = {n: t for n, t in self._tracks.items() if not t.done}
        width = max((len(n) for n in active), default=0)
        self._status.set_text("\n".join(
            f"{n:<{width}}  step {t.step:4d}" for n, t in active.items()))
