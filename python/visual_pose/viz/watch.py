"""
The standalone viewer. Tails a directory of run logs and plots whatever appears.

    .venv/bin/python -m visual_pose.viz.watch                 # out/runs
    .venv/bin/python -m visual_pose.viz.watch --latest 3 --plane xz
    .venv/bin/python -m visual_pose.viz.watch --list           # no window

Leave it open. Every time an estimator writes a run log -- from any process,
however that code has changed since -- its trajectory appears in the window and
grows as it computes. Old logs in the directory are replayed on startup, so
today's run sits next to yesterday's.

This module imports matplotlib, numpy and json. Not torch, not the descriptor
model, not the pipeline. That is deliberate: the window opens instantly and
nothing an estimator does can crash it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt

from visual_pose.viz.live_trajectory import LiveTrajectoryApp
from visual_pose.viz.run_log import DEFAULT_ROOT, SUFFIX, list_runs, summary, tail

# Tried in order when the current backend cannot open a window.
WINDOW_BACKENDS = ("macosx", "qtagg", "tkagg")


def ensure_window_backend(preferred: str | None = None) -> str:
    """Pick a backend that can open a real window, or say why not.

    A non-interactive backend makes plt.show() return at once and the process
    exits -- IDE plot panes do this and look like "made a PNG and quit".
    """
    import matplotlib

    try:
        from matplotlib.backends import BackendFilter, backend_registry
        interactive = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    except ImportError:                       # matplotlib < 3.9
        from matplotlib.rcsetup import interactive_bk as interactive

    current = matplotlib.get_backend()
    # An IDE's backend is a "module://..." name and never in that list, which is
    # the behaviour we want: it cannot hold a window open.
    if preferred is None and current.lower() in {b.lower() for b in interactive}:
        return current

    for candidate in ((preferred,) if preferred else WINDOW_BACKENDS):
        try:
            matplotlib.use(candidate, force=True)
        except Exception as exc:      # noqa: BLE001 - try the next one
            print(f"  {candidate}: {exc}")
            continue
        if preferred is None:
            print(f"backend {current!r} cannot open a window; switched to {candidate!r}")
        return candidate

    raise SystemExit(
        f"matplotlib backend {current!r} cannot open a window, and none of "
        f"{list(WINDOW_BACKENDS)} would load.\n"
        "Run this from a TERMINAL rather than the IDE's run button -- the IDE's "
        "plot pane uses a non-interactive backend that saves a PNG and exits.\n"
        "Or use --list to see the runs without a window."
    )


def play(app: LiveTrajectoryApp, path: Path, *, poll: float = 0.15) -> None:
    """Replay one log into the app, then follow it as it grows.
    """
    emitters: dict[str, object] = {}
    hints: dict[str, dict] = {}
    run_name = path.stem

    def emitter_for(track: str):
        if track not in emitters:
            hint = hints.get(track, {})
            parent = hint.get("derived_from")
            color = hint.get("color")
            if color is None and parent in emitters:
                color = app.color_of(emitters[parent].track)
            style = {"linestyle": hint["linestyle"]} if hint.get("linestyle") else {}
            emitters[track] = app.emitter(
                track, color=color, show_edges=hint.get("show_edges", False),
                **style)
        return emitters[track]

    for record in tail(path, stop=lambda: app.closed, poll=poll):
        kind = record.get("kind")
        if kind == "meta":
            run_name = record.get("name", run_name)
        elif kind == "track":
            hints[record["track"]] = record
        elif kind == "clear":
            app.clear(reference=True)
        elif kind == "reference":
            app.add_reference(record["positions"], record.get("label", "ground truth"))
        elif kind == "frame":
            emit = emitter_for(record["track"])
            if not emit.send(record["positions"], x=record.get("x"),
                             x_label=record.get("x_label", "iteration"),
                             edges=record.get("edges"),
                             keep=record.get("keep", False),
                             **record.get("scalars", {})):
                return          # window closed, or this track was removed
        elif kind == "end":
            track = record.get("track")
            if track in emitters:
                emitters[track].finish()
        elif kind == "note":
            print(f"[{run_name}] {record['text']}")

    for emit in emitters.values():
        emit.finish()           # writer died without closing; freeze what we have


def watch(app: LiveTrajectoryApp, root: Path | str = DEFAULT_ROOT, *,
          latest: int | None = None, poll: float = 0.5) -> None:
    """Tail `root`, playing each log in its own thread as it appears.
    """
    root = Path(root)

    def worker(app: LiveTrajectoryApp) -> None:
        existing = list_runs(root)
        seen = set(existing[:-latest] if latest else [])   # skipped, not replayed
        if latest and len(existing) > latest:
            print(f"skipping {len(existing) - latest} older runs "
                  f"(--latest {latest}); --all to see them")
        for path in existing:
            if path not in seen:
                seen.add(path)
                app.spawn(lambda a, p=path: play(a, p), name=p_name(path))
        while not app.closed:
            for path in list_runs(root):
                if path not in seen:
                    seen.add(path)
                    print(f"new run: {path.name}")
                    app.spawn(lambda a, p=path: play(a, p), name=p_name(path))
            time.sleep(poll)

    app.spawn(worker, name="scan")


def p_name(path: Path) -> str:
    return path.stem


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT, type=Path)
    parser.add_argument("--latest", type=int, default=4,
                        help="how many existing logs to replay (default 4)")
    parser.add_argument("--all", action="store_true", help="replay every log")
    parser.add_argument("--plane", default="3d", choices=["xy", "xz", "yz", "3d"])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--ghost-every", type=int, default=0,
                        help="keep every Nth frame as a faded line; 0 keeps none. "
                             "--ghost-every 1 shows a whole optimisation at once, "
                             "even when its iterations arrive in one burst")
    parser.add_argument("--max-ghosts", type=int, default=6,
                        help="how many faded lines to keep per track (default 6)")
    parser.add_argument("--backend", default=None,
                        help="force a matplotlib backend, e.g. macosx, qtagg, tkagg")
    parser.add_argument("--list", action="store_true",
                        help="print what is in the directory and exit")
    args = parser.parse_args(argv)

    if args.list:
        runs = list_runs(args.root)
        if not runs:
            print(f"no {SUFFIX} logs under {args.root}")
            return
        for path in runs:
            info = summary(path)
            state = "" if info["closed"] else "  (unfinished)"
            print(f"{path.name}  {info['frames']:5d} frames  "
                  f"{len(info['tracks'])} tracks  {info['meta']}{state}")
        return

    # Before the figure exists: use(force=True) closes any open one.
    backend = ensure_window_backend(args.backend)

    app = LiveTrajectoryApp(plane=args.plane, fps=args.fps,
                            ghost_every=args.ghost_every or None,
                            max_ghosts=args.max_ghosts,
                            title=f"watching {Path(args.root).name}")
    plt.ioff()                  # or show() returns immediately
    app.start(interactive=False)
    watch(app, args.root, latest=None if args.all else args.latest)
    print(f"watching {Path(args.root)} on {backend} -- q or close the window to stop")
    plt.show()

    if not app.closed:
        print(f"the window did not stay open: backend {backend!r} is not "
              "interactive here, so plt.show() returned immediately")


if __name__ == "__main__":
    main()
