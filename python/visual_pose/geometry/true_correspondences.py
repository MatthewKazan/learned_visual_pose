import numpy as np


# ─────────────── Level 1: pure geometry primitives ───────────────

def back_project(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Back-project pixels into a 3D point in the camera's frame.

    uv:    (N, 2) pixel coordinates (u, v) — u horizontal, v vertical
    depth: (N,)   depth in meters at each pixel
    K:     (3, 3) intrinsic matrix
    returns (N, 3) 3D points expressed in the camera's frame
    """
    X_C1 = (uv[:, 0] - K[0, 2]) * depth / K[0, 0]
    Y_C1 = (uv[:, 1] - K[1, 2]) * depth / K[1, 1]
    return np.stack([X_C1, Y_C1, depth], axis=1)


def project(P_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D points in a camera's frame into pixel coordinates.

    P_cam: (N, 3) 3D points
    K:     (3, 3) intrinsic matrix
    returns (N, 2) pixel coordinates (u, v)
    """
    u2 = P_cam[:, 0] * K[0, 0] / P_cam[:, 2] + K[0, 2]
    v2 = P_cam[:, 1] * K[1, 1] / P_cam[:, 2] + K[1, 2]
    return np.stack([u2, v2], axis=1)


def transform_points(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply a rigid transform to a batch of 3D points.

    P: (N, 3) 3D points
    T: (4, 4) rigid transform
    returns (N, 3) transformed points
    """
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
    Return (N, 2) integer pixel coordinates on a regular grid stepping by `step`.

    offset: (offset_u, offset_v) where the grid starts, each in [0, step).
            Callers should randomize this per sample so the model sees every
            pixel phase, not just multiples of `step`.
    """
    us, vs = np.meshgrid(np.arange(offset[0], W, step), np.arange(offset[1], H, step))
    return np.stack([us.flatten(), vs.flatten()], axis=1)   # (N, 2)



def valid_depth_mask(
    depth: np.ndarray,
    uv:    np.ndarray,
    max_depth: float = 100.0,
) -> np.ndarray:
    """
    Boolean mask over `uv` selecting pixels where depth is finite,
    positive, and less than `max_depth`.

    depth: (H, W) full depth map
    uv:    (N, 2) pixel coordinates
    returns (N,) bool
    """
    u = uv[:, 0]
    v = uv[:, 1]
    d = depth[v, u]
    return np.isfinite(d) & (d > 0) & (d < max_depth)


def in_image_mask(uv: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Boolean mask selecting pixels that lie strictly inside the image bounds.

    uv: (N, 2) — may be floats
    returns (N,) bool
    """
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
    Boolean mask: True where the depth we *predict* for uv_projected
    (= P_C_predicted's z) matches the depth actually stored at that pixel
    in frame j, within `tol` meters. Mismatch → the point is occluded.

    Assumes uv_projected is already clipped to valid image bounds; callers
    should combine with `in_image_mask` first.
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
    Sample a grid of pixels in frame i, project them into frame j via
    ground-truth depth + poses, filter invalid / off-image / occluded.

    Returns
    -------
    uv_i: (M, 2) pixels in frame i
    uv_j: (M, 2) matched pixels in frame j
    where M is the number of surviving correspondences.
    """
    H, W = depth_i.shape

    # 1. sample grid in frame i (integer pixels, guaranteed in-image)
    uv_i = sample_pixel_grid(H, W, step=sample_step, offset=grid_offset)

    # 2. back-project through frame i's depth + intrinsics → P_Ci
    d_i = depth_i[uv_i[:, 1], uv_i[:, 0]]
    P_Ci = back_project(uv_i, d_i, K)

    # 3. transform camera_i → world → camera_j → P_Cj
    T_Cj_Ci = np.linalg.inv(T_Wj) @ T_Wi # TODO: Use rotation matrix transpose instead of inverse
    P_Cj = transform_points(P_Ci, T_Cj_Ci)

    # 4. project P_Cj into frame j → uv_j (may be off-image or NaN)
    uv_j = project(P_Cj, K)

    # 5. build masks and combine
    # Clip uv_j so not_occluded_mask can safely index depth_j; out-of-image
    # pixels will get bogus occlusion values, but in_image_mask rejects them.
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

    # 6. return only surviving pixel pairs
    return uv_i[valid], uv_j[valid]