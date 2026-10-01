"""
An iPhone rosbag2 from the EECE5550 SLAM project as a FrameSequence.

    read_bag(BAGS / "inputs_20250410_203244")   -> ArraySequence

Like SceneRenderer.render_poses: decode into (rgb, depth, poses, K) arrays and
hand them to the one in-memory sequence class; nothing downstream knows it
came from a bag.

Two bag layouts, told apart by their topics:

RGB-D (app's "Upload RGB + depth" toggle, 2026-09-30 on): four topics per
frame sharing one header stamp, which is how they are paired:
    /rgbd/color/compressed   CompressedImage  jpeg 640x480
    /rgbd/color/camera_info  CameraInfo       K at 640x480
    /rgbd/depth              Image            32FC1 metres, 256x192
    /rgbd/depth/camera_info  CameraInfo       K at 256x192
Depth is resized (nearest) to the colour image so one K serves both, as with
HM3D and TartanAir. The images are landscape with +X right, +Y down, and the
depth is the camera-Z coordinate, so no axis conversion is needed. No pose is
recorded: pose() is the identity and has_ground_truth is False.

Legacy (before the toggle): /input_pointcloud only, xyz float32, 49152 points =
the 256x192 ARKit depth grid, row-major, in the CAMERA frame despite frame_id
"map": x/z is linear in the column and y/z in the row to 1e-7 (fit 2026-09-30),
fx = fy = 178.0, cx = 128.0, cy = 96.6, +Y down: OpenCV axes already, depth is
z unchanged. fx drifts 177.5-178.9 across a bag (ARKit refocuses), so K is the
per-frame fit's median. No colour: rgb() is a surface-normal image, (n + 1) / 2
per component, an experiment the descriptor was never trained for.
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import cv2
import numpy as np

from visual_pose.data_utils.dataset import ArraySequence

BAGS = Path("/Users/mattkazan/PrivateFiles/NortheasternClasses/Year5/EECE5550/FinalProject"
            "/final_project/src/slam/rosbags/input_bags")
GRID = (192, 256)     # (H, W) of the ARKit depth map

TOPIC_COLOR = "/rgbd/color/compressed"
TOPIC_COLOR_INFO = "/rgbd/color/camera_info"
TOPIC_DEPTH = "/rgbd/depth"
TOPIC_CLOUD = "/input_pointcloud"


# -- CDR ------------------------------------------------------------------------

class CdrReader:
    """
    Little-endian CDR as rosbag2 stores it: a 4-byte encapsulation header, then
    fields aligned to their own size relative to the payload start (so an 8-byte
    double sits at payload offset 8k). Strings and sequences are a u32 length
    first; a string's length counts its NUL.
    """

    def __init__(self, blob: bytes):
        self.blob = blob
        self.o = 4

    def _align(self, n: int) -> None:
        self.o = 4 + (self.o - 4 + n - 1) // n * n

    def scalar(self, fmt: str):
        n = struct.calcsize(fmt)
        self._align(n)
        v = struct.unpack_from("<" + fmt, self.blob, self.o)[0]
        self.o += n
        return v

    u8 = lambda self: self.scalar("B")
    u32 = lambda self: self.scalar("I")
    i32 = lambda self: self.scalar("i")
    f64 = lambda self: self.scalar("d")

    def string(self) -> str:
        n = self.u32()
        s = self.blob[self.o:self.o + n - 1].decode()
        self.o += n
        return s

    def array(self, dtype, count: int) -> np.ndarray:
        dtype = np.dtype(dtype)
        self._align(dtype.itemsize)
        out = np.frombuffer(self.blob, dtype=dtype.newbyteorder("<"), count=count, offset=self.o)
        self.o += count * dtype.itemsize
        return out

    def sequence(self, dtype) -> np.ndarray:
        return self.array(dtype, self.u32())

    def header(self) -> tuple[int, str]:
        """(stamp in ns, frame_id) of a std_msgs/Header."""
        sec, nsec = self.i32(), self.u32()
        return sec * 1_000_000_000 + nsec, self.string()


def parse_compressed_image(blob: bytes) -> tuple[int, np.ndarray]:
    r = CdrReader(blob)
    stamp, _ = r.header()
    r.string()                                    # format, "jpeg"
    bgr = cv2.imdecode(r.sequence(np.uint8), cv2.IMREAD_COLOR)
    return stamp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_image_32fc1(blob: bytes) -> tuple[int, np.ndarray]:
    r = CdrReader(blob)
    stamp, _ = r.header()
    h, w = r.u32(), r.u32()
    encoding = r.string()
    assert encoding == "32FC1", encoding
    r.u8()                                        # is_bigendian
    step = r.u32()
    data = r.sequence(np.uint8)
    rows = data.reshape(h, step)[:, :w * 4]
    return stamp, np.ascontiguousarray(rows).view(np.float32).reshape(h, w)


def parse_camera_info(blob: bytes) -> tuple[int, np.ndarray, tuple[int, int]]:
    """(stamp, K (3, 3), (H, W))."""
    r = CdrReader(blob)
    stamp, _ = r.header()
    h, w = r.u32(), r.u32()
    r.string()                                    # distortion_model
    r.sequence(np.float64)                        # d
    K = r.array(np.float64, 9).reshape(3, 3)
    return stamp, K.astype(np.float32), (h, w)


def parse_pointcloud_xyz(blob: bytes) -> np.ndarray:
    """(N, 3) float32 xyz from a PointCloud2 with x, y, z float32 at 0, 4, 8."""
    r = CdrReader(blob)
    r.header()
    r.u32(); r.u32()                              # height, width
    fields = []
    for _ in range(r.u32()):
        name = r.string()
        offset = r.u32()
        datatype = r.u8()
        count = r.u32()
        fields.append((name, offset, datatype, count))
    r.u8()                                        # is_bigendian
    point_step = r.u32()
    r.u32()                                       # row_step
    assert [f[:3] for f in fields] == [("x", 0, 7), ("y", 4, 7), ("z", 8, 7)], fields
    assert point_step == 12, point_step
    return r.sequence(np.uint8).view(np.float32).reshape(-1, 3)      # data is uint8[]


# -- the bag ----------------------------------------------------------------------

def open_bag(bag_dir: Path) -> sqlite3.Connection:
    return sqlite3.connect(next(Path(bag_dir).glob("*.db3")))


def topics(c: sqlite3.Connection) -> dict[str, int]:
    return {name: tid for tid, name in c.execute("select id, name from topics")}


def messages(c: sqlite3.Connection, topic_id: int) -> list[bytes]:
    return [row[0] for row in c.execute(
        "select data from messages where topic_id = ? order by timestamp", (topic_id,))]


def read_rgbd(c: sqlite3.Connection, ids: dict[str, int]):
    """
    (rgb (N, H, W, 3) uint8, depth (N, H, W) float32 at the colour size, K (3, 3)),
    for the stamps at which colour and depth both have a message; a frame
    missing either is skipped.
    """
    color = dict(parse_compressed_image(b) for b in messages(c, ids[TOPIC_COLOR]))
    depth = dict(parse_image_32fc1(b) for b in messages(c, ids[TOPIC_DEPTH]))
    infos = [parse_camera_info(b) for b in messages(c, ids[TOPIC_COLOR_INFO])]
    stamps = sorted(color.keys() & depth.keys())
    if not stamps:
        raise ValueError("no stamp has both colour and depth")
    _, K, (H, W) = infos[0]
    rgb = np.stack([color[s] for s in stamps])
    assert rgb.shape[1:3] == (H, W), (rgb.shape, (H, W))
    depth_full = np.stack([cv2.resize(depth[s], (W, H), interpolation=cv2.INTER_NEAREST)
                           for s in stamps])
    return rgb, depth_full, K


def fit_intrinsics(points: np.ndarray) -> np.ndarray:
    """
    K from a legacy cloud grid itself: x/z = (col - cx) / fx along a row and y/z
    = (row - cy) / fy down a column, least squares per frame, median over frames.
    """
    H, W = points.shape[1:3]
    fx, cx, fy, cy = [], [], [], []
    for p in points:
        u = np.nanmedian(p[..., 0] / p[..., 2], axis=0)      # (W,)
        v = np.nanmedian(p[..., 1] / p[..., 2], axis=1)      # (H,)
        a, b = np.polyfit(np.arange(W), u, 1)
        fx.append(1 / a); cx.append(-b / a)
        a, b = np.polyfit(np.arange(H), v, 1)
        fy.append(1 / a); cy.append(-b / a)
    return np.array([[np.median(fx), 0, np.median(cx)],
                     [0, np.median(fy), np.median(cy)],
                     [0, 0, 1]], dtype=np.float32)


def normal_image(points: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8: unit normal of the surface, each component mapped [-1, 1] -> [0, 255].
    Cross product of the grid's row and column tangents; the sign is fixed to face the camera."""
    d_col = np.gradient(points, axis=1)
    d_row = np.gradient(points, axis=0)
    n = np.cross(d_row, d_col)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n[n[..., 2] > 0] *= -1                          # camera looks down +Z; a visible surface faces -Z
    return ((n + 1) * 127.5).astype(np.uint8)


def read_bag(bag_dir: Path) -> ArraySequence:
    """Either layout, by its topics. 4 Hz handheld frames, so keyframes are subsampled like TartanAir."""
    with open_bag(bag_dir) as c:
        ids = topics(c)
        if TOPIC_DEPTH in ids:
            rgb, depth, K = read_rgbd(c, ids)
            depth[~np.isfinite(depth)] = 0.0             # 0 = no return, as usable_depth reads it
            return ArraySequence(rgb, depth, np.tile(np.eye(4), (len(rgb), 1, 1)), K,
                                 has_ground_truth=False)
        points = np.stack([parse_pointcloud_xyz(b).reshape(*GRID, 3)
                           for b in messages(c, ids[TOPIC_CLOUD])])
    depth = np.ascontiguousarray(points[..., 2])
    depth[~np.isfinite(depth)] = 0.0
    return ArraySequence(np.stack([normal_image(p) for p in points]), depth,
                         np.tile(np.eye(4), (len(points), 1, 1)), fit_intrinsics(points),
                         has_ground_truth=False)
