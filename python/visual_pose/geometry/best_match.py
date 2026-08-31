import torch
from torch import arange

from visual_pose.data_utils.constants import SIMILARITY_THRESHOLD

# a real peak curves downward; anything flatter than this is not something a
# parabola can localize, so refinement backs off rather than dividing by ~0
_MIN_CURVATURE = 1e-6


def unflatten_cells(flat_indices, W_feat):
    """
    Flat candidate index -> (u', v') feature-grid coordinates.

    Assumes the row-major collapse with W' on the fast axis, i.e. this is the
    inverse of permute(0,2,3,1).flatten(1,2). Kept separate from the pixel
    conversion so subpixel refinement can add a fractional offset in between.
    """
    return flat_indices % W_feat, flat_indices // W_feat


def cells_to_pixels(u_prime, v_prime, feature_space_shape, image_shape):
    """
    (u', v') feature cells -> (..., 2) image pixels, as (u, v).

    Takes floats: cell k spans stride pixels and its CENTRE is what the
    descriptor describes, which is the align_corners=False convention --
    (k + 0.5) * stride - 0.5. descriptors_at applies the matching forward
    transform; changing one without the other introduces a half-cell bias that
    cancels inside MMA but not inside an essential matrix.
    """
    assert len(feature_space_shape) == 2, "feature_space_shape must be (H, W)"
    assert len(image_shape) == 2, "image_shape must be (H, W)"

    u_pixel = (u_prime + 0.5) * (image_shape[1] / feature_space_shape[1]) - 0.5
    v_pixel = (v_prime + 0.5) * (image_shape[0] / feature_space_shape[0]) - 0.5
    return torch.stack([u_pixel, v_pixel], dim=-1)


def gridify_uvs(flat_indices, feature_space_shape, image_shape):
    """Flat index -> image pixels, no refinement. (...) -> (..., 2)"""
    u_prime, v_prime = unflatten_cells(flat_indices, feature_space_shape[1])
    return cells_to_pixels(u_prime, v_prime, feature_space_shape, image_shape)


def _parabola_offset(s_minus, s_zero, s_plus, in_bounds):
    """
    Sub-cell peak position from three samples, s_zero being the argmax.

        delta = 0.5 * (s- - s+) / (s- - 2*s0 + s+)

    Fitting a parabola through the three and solving for its vertex. Returns 0
    where the peak sits on a border (no neighbour) or the surface is too flat to
    localize, so those matches keep their integer cell rather than a wild guess.
    """
    curvature = s_minus - 2 * s_zero + s_plus
    usable = in_bounds & (curvature < -_MIN_CURVATURE)
    # substitute a harmless denominator before dividing: torch.where still
    # evaluates both branches, so a raw 0 would produce inf/nan gradients
    safe = torch.where(usable, curvature, torch.full_like(curvature, -1.0))
    delta = torch.where(usable, 0.5 * (s_minus - s_plus) / safe, torch.zeros_like(s_zero))
    return delta.clamp(-0.5, 0.5)


def subpixel_offsets(similarity, flat_indices, peak_values, feature_space_shape):
    """
    (du, dv) sub-cell corrections for each argmax, both in [-0.5, 0.5] cells.

    argmax alone quantizes every match to the stride -- 8 px here -- and that
    quantization is correlated across matches (everything lands on the same
    lattice), so it biases a pose fit rather than averaging out. Measured cost
    with otherwise-perfect correspondences: ~5.9 deg of translation direction.

    The four neighbours are a fixed offset away in the flat layout, so this
    needs gathers rather than reshaping the (B, N, M) similarity matrix.
    """
    H_feat, W_feat = feature_space_shape
    n_candidates = H_feat * W_feat
    u_prime, v_prime = unflatten_cells(flat_indices, W_feat)

    def neighbour(offset):
        # clamp keeps the gather in range; the in_bounds masks below discard
        # whatever the clamped index happened to read
        safe_index = (flat_indices + offset).clamp(0, n_candidates - 1)
        return similarity.gather(2, safe_index.unsqueeze(-1)).squeeze(-1)

    du = _parabola_offset(neighbour(-1), peak_values, neighbour(1),
                          (u_prime > 0) & (u_prime < W_feat - 1))
    dv = _parabola_offset(neighbour(-W_feat), peak_values, neighbour(W_feat),
                          (v_prime > 0) & (v_prime < H_feat - 1))
    return du, dv


def best_match(
        descriptors_j: torch.Tensor,
        image_shape: tuple[int, int],
        query_descriptors_i: torch.Tensor,
        subpixel: bool = True,
):
    """
    Take a grid of descriptors for each image, and find the best matches.

    :param query_descriptors_i: (B, N, D) L2-normalized query descriptors for image i --
                          DescriptorCNN.forward output flattened from (B, D, H, W)
    :param descriptors_j: (B, D, H', W') L2-normalized descriptors for image j
    :param image_shape:   (H, W) of the original image. H / H' is the total
                          stride, needed to map feature cells back to pixels
    :param subpixel:      refine each argmax against its neighbours. Off gives
                          the raw lattice, which is what MMA@8 was measured on.

    :return: (uv_j, score, mutual)
             uv_j   (B, N, 2) float, best match per query in image j pixel coords
             score  (B, N)    the winning cosine -- the rejection threshold reads this
             mutual (B, N)    bool, True where the match points back at its query
    """
    assert len(image_shape) == 2, "image_shape must be (H, W)"
    feature_space_shape = tuple(descriptors_j.shape[2:])

    # the answer pool: image j's whole map as a flat list of candidates.
    # (B,D,H',W') -> (B,H',W',D) -> (B, M, D) with M = H'*W'.
    # Row-major collapse puts W' on the fast axis, hence the div/mod below.
    candidate_j = descriptors_j.permute(0, 2, 3, 1).flatten(1, 2)

    # (B,N,D) @ (B,D,M) -> (B, N, M). Descriptors are unit length, so the
    # dot product IS the cosine. argmax over M -> (B, N) winning indices.
    similarity = torch.matmul(query_descriptors_i, candidate_j.transpose(-1, -2))
    best = similarity.max(dim=-1)  # best.values (B,N), best.indices (B,N)

    # injectivity: the raw argmax will happily map many queries onto one cell.
    # "who does my match point back at -- me?"
    nn_ji = similarity.argmax(dim=1)  # (B, M) each candidate's best query
    queries = arange(query_descriptors_i.shape[1], device=similarity.device)
    mutual = nn_ji.gather(1, best.indices) == queries  # (B, N) bool

    u_prime, v_prime = unflatten_cells(best.indices, feature_space_shape[1])
    if subpixel:
        du, dv = subpixel_offsets(similarity, best.indices, best.values, feature_space_shape)
        u_prime, v_prime = u_prime + du, v_prime + dv

    uv_j = cells_to_pixels(u_prime, v_prime, feature_space_shape, image_shape)
    return uv_j, best.values, mutual


def get_matching_pairs(model, images_i, images_j):
    assert images_i.shape == images_j.shape, "images_i and images_j must have the same shape"
    # boolean masking below flattens the batch, so B>1 would silently merge
    # correspondences from different image pairs into one list -- and a single
    # essential matrix fitted across two camera motions is meaningless.
    assert images_i.shape[0] == 1, "only batch size 1 is supported"

    descriptors_i = model(images_i)
    descriptors_j = model(images_j)

    uv_i = gridify_uvs(
        flat_indices=arange(descriptors_i.shape[2] * descriptors_i.shape[3],
                            device=descriptors_i.device),
        feature_space_shape=descriptors_i.shape[2:],
        image_shape=images_i.shape[2:]
    )
    # every batch element queries the same grid, so this is a stride-0 view
    uv_i = uv_i.expand(len(images_i), -1, -1)

    query_descriptors = descriptors_i.permute(0, 2, 3, 1).flatten(1, 2)

    best_uv_js, scores, mutual = best_match(
        query_descriptors_i=query_descriptors,
        descriptors_j=descriptors_j,
        image_shape=(images_j.shape[2], images_j.shape[3])
    )
    # check alignment BEFORE masking -- afterwards both sides carry the same
    # mask and the comparison is a tautology
    assert best_uv_js.shape == uv_i.shape, "mismatch in shape between query descriptors and best matches"

    mask = mutual & (scores > SIMILARITY_THRESHOLD)
    return uv_i[mask], best_uv_js[mask]
