import torch
from roma import rotvec_to_rotmat, rotmat_to_rotvec
from torch import Tensor


def relative(R: Tensor, t: Tensor, ij: Tensor) -> tuple[Tensor, Tensor]:
    """
    Per edge, pose of j seen from i, as Z_ij = T_i^-1 T_j: R_i^T R_j and R_i^T (t_j - t_i).
    R (B, V, 3, 3), t (B, V, 3), ij (B, E, 2) -> (B, E, 3, 3), (B, E, 3). Padded edges index
    vertex 0 at both ends: identity, finite, masked by the caller.
    """
    b = torch.arange(R.shape[0], device=R.device)[:, None]           # (B, 1): each graph indexes its own vertices
    i, j = ij[..., 0], ij[..., 1]                                    # (B, E)
    Ri_T = R[b, i].transpose(-1, -2)                                 # (B, E, 3, 3)
    return Ri_T @ R[b, j], (Ri_T @ (t[b, j] - t[b, i]).unsqueeze(-1)).squeeze(-1)


def predicted_poses(predicted_delta: Tensor, poses: Tensor) -> tuple[Tensor, Tensor]:
    """
    Each vertex's corrected pose in the vertex-0 frame, R = R_in Exp(phi), t = t_in + R_in rho --
    the same rule as evaluation's correction(). predicted_delta (B, V, 6) [rho, phi]
    -> R (B, V, 3, 3), t (B, V, 3).
    """
    pose_mat = poses.unflatten(-1, (3, 4))                  # (B, V, 12) -> (B, V, 3, 4): R_in | t_in
    R_in, t_in = pose_mat[..., :3, :3], pose_mat[..., :3, 3]         # (B, V, 3, 3), (B, V, 3)
    R = R_in @ rotvec_to_rotmat(predicted_delta[..., 3:])            # (B, V, 3, 3)
    t = t_in + (R_in @ predicted_delta[..., :3].unsqueeze(-1)).squeeze(-1)  # (B, V, 3)
    return R, t


def update_graph_with_tangent_delta(poses: Tensor, predicted_delta: Tensor) -> Tensor:
    """
    Given a corrective vector in tangent space of SE3, update the pose.

    :param poses: The current best guess of the pose. (B, V, 12)
    :param predicted_delta: The corrective vector in tangent space of SE3. (B, V, 6)

    :return: The updated poses, and edges.
    """
    pred_R, pred_t = predicted_poses(predicted_delta, poses)
    pred_poses = torch.cat((pred_R, pred_t.unsqueeze(-1)), dim=-1).flatten(-2, -1)

    return pred_poses


def update_ground_truth_residual(updated_poses: Tensor, gt_poses: Tensor) -> tuple[Tensor, Tensor]:
    """

    :param updated_poses:
    :param gt_poses:
    :return:
    """
    pose_mat = updated_poses.unflatten(-1, (3, 4))                  # (B, V, 12) -> (B, V, 3, 4): R_in | t_in
    R_in, t_in = pose_mat[..., :3, :3], pose_mat[..., :3, 3]         # (B, V, 3, 3), (B, V, 3)
    gt_R, gt_t = gt_poses[..., :3, :3], gt_poses[..., :3, 3]

    residuals_R = R_in.transpose(-1, -2) @ gt_R
    residuals_t = (R_in.transpose(-1, -2) @ (gt_t - t_in).unsqueeze(-1))
    residual_log = torch.cat((residuals_t.squeeze(-1), rotmat_to_rotvec(residuals_R)), dim=-1)  # [rho, phi]
    return torch.cat((residuals_R, residuals_t), dim=-1), residual_log






