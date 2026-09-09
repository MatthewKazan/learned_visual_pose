import numpy as np


# ─────────────── Level 1: pure geometry primitives ───────────────

def back_project(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """uv (N,2) pixels + depth (N,) metres + K (3,3) -> (N,3) camera-frame points."""
    X_C1 = (uv[:, 0] - K[0, 2]) * depth / K[0, 0]
    Y_C1 = (uv[:, 1] - K[1, 2]) * depth / K[1, 1]
    return np.stack([X_C1, Y_C1, depth], axis=1)


def project(P_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """P_cam (N,3) camera-frame points + K (3,3) -> (N,2) pixels."""
    u2 = P_cam[:, 0] * K[0, 0] / P_cam[:, 2] + K[0, 2]
    v2 = P_cam[:, 1] * K[1, 1] / P_cam[:, 2] + K[1, 2]
    return np.stack([u2, v2], axis=1)


def transform_points(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """P (N,3) points through T (4,4) -> (N,3)."""
    R = T[:3, :3]
    t = T[:3, 3]
    return P @ R.T + t


# ─────────────── Level 2: sampling and validity ───────────────

def sample_pixel_grid(
    H: int,
    W: int,
    step: int = 8,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    (N, 2) integer pixels on a grid of spacing `step`.

    offset is where the grid starts, each component in [0, step). Randomise it
    per sample or the model only ever sees multiples of `step`.
    """
    us, vs = np.meshgrid(np.arange(offset[0], W, step), np.arange(offset[1], H, step))
    return np.stack([us.flatten(), vs.flatten()], axis=1)   # (N, 2)



def depth_at(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    Bilinear depth at subpixel uv. depth (H,W), uv (N,2) as (x,y) -> (N,).

    Rounding costs |t| accuracy directly: p50 0.74 -> 0.62 cm against nearest
    neighbour, worst edge 3.42 -> 2.00 deg.
    """
    u0 = np.floor(uv[:, 0]).astype(int)
    v0 = np.floor(uv[:, 1]).astype(int)

    u1 = np.clip(u0 + 1, 0, depth.shape[1] - 1)
    v1 = np.clip(v0 + 1, 0, depth.shape[0] - 1)
    u0 = np.clip(u0, 0, depth.shape[1] - 1)
    v0 = np.clip(v0, 0, depth.shape[0] - 1)

    fractional_u = uv[:, 0] - u0
    fractional_v = uv[:, 1] - v0

    w1 = (1 - fractional_u) * (1 - fractional_v) * depth[v0, u0]
    w2 = (1 - fractional_u) * fractional_v * depth[v1, u0]
    w3 = fractional_u * (1 - fractional_v) * depth[v0, u1]
    w4 = fractional_u * fractional_v * depth[v1, u1]
    return w1 + w2 + w3 + w4


def usable_depth(d: np.ndarray, max_depth: float = 100.0) -> np.ndarray:
    """
    (N,) bool over sampled depths: finite, positive, under max_depth.

    0 is TartanAir's no-return code and past max_depth is sky. Back-projecting
    either invents a point no estimator can reject from its residual.
    """
    return np.isfinite(d) & (d > 0) & (d < max_depth)


def valid_depth_mask(
    depth: np.ndarray,
    uv:    np.ndarray,
    max_depth: float = 100.0,
) -> np.ndarray:
    """
    usable_depth at INTEGER uv. (N,) bool.

    It indexes the map, so subpixel uv must go through depth_at instead.
    """
    return usable_depth(depth[uv[:, 1], uv[:, 0]], max_depth)


def in_image_mask(uv: np.ndarray, H: int, W: int) -> np.ndarray:
    """(N,) bool: uv (N,2), possibly float, strictly inside the image."""
    u = uv[:, 0]
    v = uv[:, 1]
    return (u >= 0) & (u < W) & (v >= 0) & (v < H)


def not_occluded_mask(
    P_C_predicted: np.ndarray,   # (N, 3) 3D points in camera j's frame
    uv_projected:  np.ndarray,   # (N, 2) pixel coords in frame j
    depth_j:       np.ndarray,   # (H, W) depth map of frame j
    tol:           float = 0.5,  # meters
) -> np.ndarray:
    """
    (N,) bool: predicted depth matches frame j's measured depth within tol.

    A mismatch means something else is in front, so the point is occluded.
    uv_projected must already be clipped in-bounds; combine with in_image_mask.
    """

    return np.abs(depth_j[uv_projected[:, 1], uv_projected[:, 0]] - P_C_predicted[:, 2]) < tol


# ─────────────── Level 3: the composed function ───────────────

def generate_correspondences(
    depth_i:       np.ndarray,   # (H, W) frame i's depth map
    depth_j:       np.ndarray,   # (H, W) frame j's depth map
    T_Wi:          np.ndarray,   # (4, 4) world <- camera i
    T_Wj:          np.ndarray,   # (4, 4) world <- camera j
    K:             np.ndarray,   # (3, 3) intrinsics (both cameras identical)
    sample_step:   int   = 8,
    grid_offset:   tuple[int, int] = (0, 0),
    max_depth:     float = 100.0,
    occlusion_tol: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Grid pixels in frame i, pushed into frame j through depth and poses.

    Drops invalid depth, points behind camera j, off-image, and occluded.
    Returns (uv_i, uv_j), (M, 2) each.
    """
    H, W = depth_i.shape

    # integer pixels, so in-image by construction
    uv_i = sample_pixel_grid(H, W, step=sample_step, offset=grid_offset)

    d_i = depth_i[uv_i[:, 1], uv_i[:, 0]]
    P_Ci = back_project(uv_i, d_i, K)

    T_Cj_Ci = np.linalg.inv(T_Wj) @ T_Wi # TODO: Use rotation matrix transpose instead of inverse
    P_Cj = transform_points(P_Ci, T_Cj_Ci)

    uv_j = project(P_Cj, K)

    # clip so not_occluded_mask can index safely; in_image_mask rejects the
    # out-of-bounds ones anyway
    uv_j_safe = np.stack([
        np.clip(uv_j[:, 0], 0, W - 1).astype(int),
        np.clip(uv_j[:, 1], 0, H - 1).astype(int),
    ], axis=1)

    valid = (
        valid_depth_mask(depth_i, uv_i, max_depth=max_depth)
        & (P_Cj[:, 2] > 0)
        & in_image_mask(uv_j, H, W)
        & not_occluded_mask(P_Cj, uv_j_safe, depth_j, tol=occlusion_tol)
    )

    return uv_i[valid], uv_j[valid]