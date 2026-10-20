"""
The viewer's File > Open, turned into "add to rerun_app's SHOW".

The viewer runs every `rerun-importer-*` on its PATH for each file opened; bin/
holds ours, and rerun_app puts bin/ on the PATH of the viewer it spawns. Draws
nothing: appends the file's globs to REQUESTS and exits "not mine", so the
scene arrives through rerun_app as usual (runs, frames on click, eviction).

Opens out/viewer/open/*.show: one per scene, written by rerun_app, each line a
SHOW glob; select several, or write your own. Not scene folders (opening one
also hands every png and the .glb to rerun's own importers), and not poses.txt
(rerun never passes .txt to external importers; tested 0.38.1).

Stdlib only, no visual_pose: it runs once per file opened.
"""
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3]
HANDLES = REPO_DIR / "out" / "viewer" / "open"
REQUESTS = REPO_DIR / "out" / "viewer" / "requests.txt"
HANDLE_SUFFIX = ".show"
NOT_MINE = 66      # rerun.EXTERNAL_IMPORTER_INCOMPATIBLE_EXIT_CODE


def patterns(path: Path) -> list[str]:
    if path.suffix != HANDLE_SUFFIX:
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    path = Path(sys.argv[1])
    if path.is_file() and (found := patterns(path)):
        REQUESTS.parent.mkdir(parents=True, exist_ok=True)
        with REQUESTS.open("a") as f:
            f.write("".join(p + "\n" for p in found))
    sys.exit(NOT_MINE)


if __name__ == "__main__":
    main()
