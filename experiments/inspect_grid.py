"""Show every grid point on frame i, coloured by the first filter it failed.

    python -m experiments.inspect_grid

Answers "why so few correspondences?" -- the ceiling is (H/step)*(W/step), and
outdoor frames lose most of it to sky (depth 65504m) while indoor-shadow frames
lose it to the photometric filter. Writes a PNG rather than calling plt.show(),
which blocks on a GUI backend.
"""

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, torch
from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.data_utils.constants import REPO_DIR, INTRINSICS_TARTAN_AIR as K
from visual_pose.geometry.true_correspondences import (
    sample_pixel_grid, back_project, transform_points, project,
    valid_depth_mask, in_image_mask, not_occluded_mask)

root = REPO_DIR/'data/tartan_air'
cases = [('P002', 0), ('P002', 133), ('P000', 100)]
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

for ax, (name, i) in zip(axes, cases):
    s = TartanAirSequence(root/name); j = i + 5
    di, dj = s.depth(i), s.depth(j)
    H, W = di.shape
    uv = sample_pixel_grid(H, W, 8)
    P_Ci = back_project(uv, di[uv[:,1], uv[:,0]], K)
    P_Cj = transform_points(P_Ci, np.linalg.inv(s.pose(j)) @ s.pose(i))
    uvj = project(P_Cj, K)
    uvs = np.stack([np.clip(uvj[:,0],0,W-1).astype(int), np.clip(uvj[:,1],0,H-1).astype(int)], 1)

    m_depth = valid_depth_mask(di, uv)
    m_z     = P_Cj[:,2] > 0
    m_img   = in_image_mask(uvj, H, W)
    m_occ   = not_occluded_mask(P_Cj, uvs, dj)
    ii = torch.from_numpy(s.rgb(i)).permute(2,0,1).float()/255.
    m_dark  = ii[:, uv[:,1], uv[:,0]].mean(0).numpy() >= 0.05

    # first failing stage per point, in pipeline order
    fate = np.full(len(uv), 5)
    for k, m in enumerate([m_depth, m_z, m_img, m_occ, m_dark]):
        fate[(fate == 5) & ~m] = k

    ax.imshow(s.rgb(i))
    labels = ['depth invalid / sky', 'behind camera j', 'off-image in j', 'occluded', 'too dark', 'KEPT']
    colors = ['#ef4444', '#f97316', '#eab308', '#a855f7', '#3b82f6', '#22c55e']
    for k in range(6):
        sel = fate == k
        if sel.sum():
            ax.plot(uv[sel,0], uv[sel,1], '.', ms=2.5, color=colors[k],
                    label=f'{labels[k]} {sel.sum()/len(uv):.0%}')
    ax.set_title(f'{name} frame {i}->{j}   {(fate==5).sum()}/{len(uv)} kept', fontsize=10)
    ax.legend(loc='upper right', fontsize=7, markerscale=4, framealpha=0.9)
    ax.axis('off')

fig.tight_layout(); fig.savefig(str(REPO_DIR / 'experiments' / 'grid.png'), dpi=95, bbox_inches='tight')
print('ok')
