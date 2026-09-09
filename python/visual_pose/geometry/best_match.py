import torch
from torch import Tensor, arange, nn

# flatter than this and the parabola fit divides by ~0, so refinement backs off
_MIN_CURVATURE = 1e-6


def unflatten_cells(flat_indices: Tensor, W_feat: int) -> tuple[Tensor, Tensor]:
    """
    Flat candidate index -> (u', v') feature cells.

    Inverse of permute(0,2,3,1).flatten(1,2), so W' is the fast axis. Separate
    from the pixel conversion so refinement can add a fraction in between.
    """
    return flat_indices % W_feat, flat_indices // W_feat


def cells_to_pixels(u_prime: Tensor, v_prime: Tensor,
                    feature_space_shape: tuple[int, int],
                    image_shape: tuple[int, int]) -> Tensor:
    """
    (u', v') feature cells -> (..., 2) image pixels. Floats allowed.

        pixel = (cell + 0.5) * stride - 0.5

    The descriptor describes the cell CENTRE, hence the half-cell terms.
    descriptors_at does the forward version; change one without the other and
    you get a half-cell bias that MMA hides but a pose estimate does not.
    """
    assert len(feature_space_shape) == 2, "feature_space_shape must be (H, W)"
    assert len(image_shape) == 2, "image_shape must be (H, W)"

    u_pixel = (u_prime + 0.5) * (image_shape[1] / feature_space_shape[1]) - 0.5
    v_pixel = (v_prime + 0.5) * (image_shape[0] / feature_space_shape[0]) - 0.5
    return torch.stack([u_pixel, v_pixel], dim=-1)


def gridify_uvs(flat_indices: Tensor, feature_space_shape: tuple[int, int],
                image_shape: tuple[int, int]) -> Tensor:
    """Flat index -> image pixels, no refinement. (...) -> (..., 2)"""
    u_prime, v_prime = unflatten_cells(flat_indices, feature_space_shape[1])
    return cells_to_pixels(u_prime, v_prime, feature_space_shape, image_shape)


def _parabola_offset(s_minus: Tensor, s_zero: Tensor, s_plus: Tensor,
                     in_bounds: Tensor) -> Tensor:
    """
    Sub-cell peak from three samples, s_zero being the argmax.

        delta = 0.5 * (s- - s+) / (s- - 2*s0 + s+)

    A parabola through the three, solved for its vertex. Returns 0 on a border
    (no neighbour) or a too-flat peak, so those keep their integer cell.
    """
    curvature = s_minus - 2 * s_zero + s_plus
    usable = in_bounds & (curvature < -_MIN_CURVATURE)
    # substitute a harmless denominator before dividing: torch.where still
    # evaluates both branches, so a raw 0 would produce inf/nan gradients
    safe = torch.where(usable, curvature, torch.full_like(curvature, -1.0))
    delta = torch.where(usable, 0.5 * (s_minus - s_plus) / safe, torch.zeros_like(s_zero))
    return delta.clamp(-0.5, 0.5)


def subpixel_offsets(similarity: Tensor, flat_indices: Tensor, peak_values: Tensor,
                     feature_space_shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """
    (du, dv) sub-cell corrections per argmax, each in [-0.5, 0.5] cells.

    similarity (B, N, M), flat_indices and peak_values (B, N) -> (B, N) each.

    Without this every match lands on the stride lattice, and that error is the
    same for all of them, so it biases a pose fit instead of averaging out --
    ~5.9 deg of translation direction on otherwise-perfect correspondences.

    Neighbours are a fixed offset in the flat layout, hence gathers rather than
    reshaping the (B, N, M) matrix.
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
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Nearest neighbour in image j for each query descriptor from image i.

    :param query_descriptors_i: (B, N, D) unit-length queries
    :param descriptors_j:       (B, D, H', W') unit-length map to search
    :param image_shape:         (H, W); H/H' is the stride
    :param subpixel:            refine each argmax against its neighbours

    :return: uv_j   (B, N, 2) match position in image j pixels
             score  (B, N) winning cosine, what the threshold reads
             mutual (B, N) True where the match points back at its query
    """
    assert len(image_shape) == 2, "image_shape must be (H, W)"
    feature_space_shape = tuple(descriptors_j.shape[2:])

    # image j's whole map as M = H'*W' candidates. Row-major, so W' is the
    # fast axis -- which is what unflatten_cells assumes.
    candidate_j = descriptors_j.permute(0, 2, 3, 1).flatten(1, 2)

    # unit length, so the dot product is the cosine. (B, N, M)
    similarity = torch.matmul(query_descriptors_i, candidate_j.transpose(-1, -2))
    best = similarity.max(dim=-1)  # best.values (B,N), best.indices (B,N)

    # raw argmax maps many queries onto one cell, so ask each winner who its
    # own best query is
    nn_ji = similarity.argmax(dim=1)  # (B, M) each candidate's best query
    queries = arange(query_descriptors_i.shape[1], device=similarity.device)
    mutual = nn_ji.gather(1, best.indices) == queries  # (B, N) bool

    u_prime, v_prime = unflatten_cells(best.indices, feature_space_shape[1])
    if subpixel:
        du, dv = subpixel_offsets(similarity, best.indices, best.values, feature_space_shape)
        u_prime, v_prime = u_prime + du, v_prime + dv

    uv_j = cells_to_pixels(u_prime, v_prime, feature_space_shape, image_shape)
    return uv_j, best.values, mutual


def get_matching_pairs(model: nn.Module, images_i: Tensor,
                       images_j: Tensor, similarity_threshold: float) -> tuple[Tensor, Tensor]:
    """
    Mutual, above-threshold correspondences for one pair. (B, 3, H, W) each
    -> (uv_i, uv_j), (K, 2) image pixels each. K varies with the pair.
    """
    assert images_i.shape == images_j.shape, "images_i and images_j must have the same shape"
    # the mask below flattens the batch, so B>1 would merge correspondences
    # from different pairs into one list
    assert images_i.shape[0] == 1, "only batch size 1 is supported"

    descriptors_i = model(images_i)
    descriptors_j = model(images_j)
    return get_matching_pairs_from_descriptors(
        descriptors_i, descriptors_j, tuple(images_i.shape[2:]), similarity_threshold)



def get_matching_pairs_from_descriptors(descriptors_i, descriptors_j, image_shape,
                                        similarity_threshold):
    """Same as get_matching_pairs but on already-encoded descriptor maps."""
    uv_i = gridify_uvs(
        flat_indices=arange(descriptors_i.shape[2] * descriptors_i.shape[3],
                            device=descriptors_i.device),
        feature_space_shape=descriptors_i.shape[2:],
        image_shape=image_shape
    )
    # every batch element queries the same grid, so this is a stride-0 view
    uv_i = uv_i.expand(1, -1, -1)

    query_descriptors = descriptors_i.permute(0, 2, 3, 1).flatten(1, 2)

    best_uv_js, scores, mutual = best_match(
        query_descriptors_i=query_descriptors,
        descriptors_j=descriptors_j,
        image_shape=image_shape
    )
    # before masking: afterwards both sides carry the same mask and this is
    # a tautology
    assert best_uv_js.shape == uv_i.shape, "mismatch in shape between query descriptors and best matches"

    mask = mutual & (scores > similarity_threshold)
    return uv_i[mask], best_uv_js[mask]
