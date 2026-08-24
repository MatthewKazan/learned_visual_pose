import matplotlib.pyplot as plt
from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.data_utils.constants import REPO_DIR, INTRINSICS_TARTAN_AIR
from visual_pose.geometry.correspondences import (
    generate_correspondences
)


def draw_correspondences(rgb_i, rgb_j, uv_i, uv_j, subsample=100):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
    a.imshow(rgb_i); a.set_title("frame i")
    b.imshow(rgb_j); b.set_title("frame j")
    for k in range(0, len(uv_i)):
        a.plot(*uv_i[k], "ro", markersize=3)
        b.plot(*uv_j[k], "ro", markersize=3)
    plt.tight_layout()
    plt.show()


def main():
    seq = TartanAirSequence(REPO_DIR / "data" / "tartan_air" / "P002")
    i, j = 0, 5

    uv_i, uv_j = generate_correspondences(
        depth_i = seq.depth(i),
        depth_j = seq.depth(j),
        T_Wi    = seq.pose(i),
        T_Wj    = seq.pose(j),
        K       = INTRINSICS_TARTAN_AIR,
    )

    print(f"{len(uv_i)} valid correspondences between frames")
    draw_correspondences(seq.rgb(i), seq.rgb(j), uv_i, uv_j)


if __name__ == "__main__":
    main()