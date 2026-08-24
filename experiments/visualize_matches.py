"""
Visual check on descriptor quality: untrained vs trained, side by side.

    python -m experiments.visualize_matches
    python -m experiments.visualize_matches --checkpoint checkpoints/last.pt --pair 120

Three views, each answering a different question:

  1. MATCH LINES  -- for a handful of ground-truth correspondences, draw a line
     from the query pixel in image i to where the descriptor says it went in
     image j. Green = within tau px of truth, red = wrong. Reading: a good
     model gives mostly-green lines that are roughly PARALLEL, because true
     correspondences across a small camera motion all shift in a similar
     direction. Red lines scattered in every direction = matching noise.

  2. SIMILARITY HEATMAP -- pick ONE query pixel and show its cosine similarity
     against every location in image j. Reading: a good descriptor gives a
     single sharp peak on the true location. A diffuse blob means the
     descriptor can't localize; several bright peaks mean it can't tell
     repeated structure apart; uniform brightness means collapse.

  3. DESCRIPTOR PCA -- project the 128-D descriptor at every location down to
     3 dims and render it as RGB. Both images use the SAME basis, so the same
     surface should get the same colour in both. Reading: coherent regions
     that agree across the two views = descriptors encode scene structure. One
     flat colour = collapsed. Pure noise = no spatial structure learned.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.data_utils.training_dataset import TACorrespondenceDataset
from visual_pose.models.descriptor_cnn import DescriptorCNN
from visual_pose.models.sift import SIFT
from visual_pose.models.training import descriptors_at, mma

TAU = 8.0          # "correct" threshold in pixels, matches the stride
N_LINES = 40       # correspondences to draw; more than this is unreadable


def most_textured_pair(seq, dataset, stride=10):
    """
    Pick the pair whose first frame has the most texture, by Laplacian variance.

    Worth automating: a blank-wall frame gives no descriptor anything to work
    with, so it makes a trained and an untrained model look equally bad. Texture
    across this sequence spans a 10x range, so an arbitrary index is a coin flip
    on whether the figure shows you anything.
    """
    best_score, best_idx = -1.0, 0
    for pair_idx in range(0, len(dataset), stride):
        frame_i = dataset.pairs[pair_idx][0]
        grey = cv2.cvtColor(seq.rgb(frame_i), cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(grey, cv2.CV_64F).var()
        if score > best_score:
            best_score, best_idx = score, pair_idx
    return best_idx


def load_model(checkpoint=None):
    """
    Build the architecture recorded beside the checkpoint, not DescriptorCNN's
    defaults -- a kernel-7 checkpoint will not load into a kernel-3 model, and
    the sweep moved the architecture well away from the class defaults.
    """
    cfg = Config()
    if checkpoint is not None:
        cfg_path = Path(checkpoint).parent / "config.json"
        if cfg_path.exists():
            saved = json.loads(cfg_path.read_text())
            cfg = Config(**{k: tuple(v) if isinstance(v, list) else v
                            for k, v in saved.items()})

    model = DescriptorCNN(
        body_channels=list(cfg.body_channels),
        body_kernel_sizes=list(cfg.body_kernel_sizes),
        body_strides=list(cfg.body_strides),
        body_dilations=list(cfg.body_dilations),
        descriptor_dim=cfg.descriptor_dim,
        norm=cfg.norm,
    ).to(DEVICE).eval()

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"loaded {checkpoint}  (epoch {state['epoch'] + 1}, "
              f"val_mma {state['val_mma']:.2%}, kernels {cfg.body_kernel_sizes} "
              f"dilations {cfg.body_dilations})")
    return model


@torch.no_grad()
def analyse(model, batch):
    """Everything the plots need, for one batch of size 1."""
    images_i, images_j = batch["rgb_i"], batch["rgb_j"]
    uvs_i, uvs_j = batch["uv_i"], batch["uv_j"]

    queries, _ = descriptors_at(model, images_i, uvs_i)      # (1, N, D)
    feature_j = model(images_j)                              # (1, D, H', W')
    Hf, Wf = feature_j.shape[2], feature_j.shape[3]
    candidates = feature_j.permute(0, 2, 3, 1).flatten(1, 2)  # (1, M, D)

    similarity = torch.matmul(queries, candidates.transpose(-1, -2))  # (1, N, M)
    best = similarity.argmax(dim=-1)                                  # (1, N)

    stride_u = images_j.shape[3] / Wf
    stride_v = images_j.shape[2] / Hf
    predicted = torch.stack([(best % Wf) * stride_u, (best // Wf) * stride_v], dim=-1)
    error = torch.linalg.vector_norm(predicted - uvs_j, dim=-1)        # (1, N)

    return {
        "uv_i": uvs_i[0].cpu(),
        "uv_j": uvs_j[0].cpu(),
        "predicted": predicted[0].cpu(),
        "error": error[0].cpu(),
        "similarity": similarity[0].cpu(),   # (N, M)
        "feature_i": model(images_i)[0].cpu(),
        "feature_j": feature_j[0].cpu(),
        "shape": (Hf, Wf),
    }


def to_numpy_image(t):
    return t[0].permute(1, 2, 0).cpu().numpy()


def draw_match_lines(ax, img_i, img_j, res, title):
    """Side-by-side images with lines from query to predicted match."""
    H, W = img_i.shape[:2]
    ax.imshow(np.concatenate([img_i, img_j], axis=1))

    idx = np.linspace(0, len(res["uv_i"]) - 1, N_LINES).astype(int)
    n_ok = 0
    for n in idx:
        u0, v0 = res["uv_i"][n]
        u1, v1 = res["predicted"][n]
        ok = res["error"][n] < TAU
        n_ok += int(ok)
        ax.plot([u0, u1 + W], [v0, v1], lw=0.8,
                color="#22c55e" if ok else "#ef4444", alpha=0.85)

    overall = (res["error"] < TAU).float().mean()
    ax.set_title(f"{title}\nmatches drawn: {n_ok}/{len(idx)} correct   |   "
                 f"all {len(res['error'])}: MMA@{TAU:.0f} = {overall:.1%}", fontsize=9)
    ax.axis("off")


def draw_heatmap(ax, img_j, res, query_n, title):
    """Similarity of ONE query against every location in image j."""
    Hf, Wf = res["shape"]
    heat = res["similarity"][query_n].reshape(Hf, Wf)[None, None]
    heat = F.interpolate(heat, size=img_j.shape[:2], mode="bilinear", align_corners=False)

    ax.imshow(img_j)
    im = ax.imshow(heat[0, 0].numpy(), cmap="inferno", alpha=0.72)
    tu, tv = res["uv_j"][query_n]
    pu, pv = res["predicted"][query_n]
    ax.plot(tu, tv, "o", mfc="none", mec="#22c55e", ms=16, mew=2.2, label="truth")
    ax.plot(pu, pv, "x", color="#38bdf8", ms=13, mew=2.5, label="argmax")

    peak = res["similarity"][query_n].max().item()
    mean = res["similarity"][query_n].mean().item()
    ax.set_title(f"{title}\npeak {peak:.2f}  mean {mean:.2f}  "
                 f"contrast {peak - mean:.2f}  err {res['error'][query_n]:.1f}px", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax.axis("off")
    return im


def draw_pca(ax, res, title):
    """
    Both feature maps projected onto ONE shared 3-D basis, rendered as RGB.
    Shared basis is the point: it makes colours comparable across the two views.
    """
    D, Hf, Wf = res["feature_i"].shape
    fi = res["feature_i"].permute(1, 2, 0).reshape(-1, D)
    fj = res["feature_j"].permute(1, 2, 0).reshape(-1, D)
    both = torch.cat([fi, fj], dim=0)

    both = both - both.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(both, q=3)
    proj = both @ V[:, :3]

    # measure explained variance BEFORE the min-max rescale below, which would
    # otherwise destroy the scale the ratio depends on
    explained = (proj.var(dim=0).sum() / both.var(dim=0).sum()).item()

    proj = (proj - proj.min(0).values) / (proj.max(0).values - proj.min(0).values + 1e-8)
    rgb_i = proj[: Hf * Wf].reshape(Hf, Wf, 3).numpy()
    rgb_j = proj[Hf * Wf:].reshape(Hf, Wf, 3).numpy()
    ax.imshow(np.concatenate([rgb_i, rgb_j], axis=1))
    ax.set_title(f"{title}\ndescriptor PCA -> RGB (view i | view j), "
                 f"top-3 dims hold {explained:.1%} of variance", fontsize=9)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO_DIR / "checkpoints" / "best.pt"))
    ap.add_argument("--sequence", default="P003", help="held-out sequence to visualise")
    ap.add_argument("--pair", type=int, default=None,
                    help="which frame pair (default: auto-pick the most textured one)")
    ap.add_argument("--query", type=int, default=None,
                    help="which correspondence to heatmap (default: a mid-image one)")
    ap.add_argument("--out", default=str(REPO_DIR / "experiments" / "matches.png"))
    ap.add_argument("--models", default="untrained,trained",
                    help="comma-separated columns from: sift, untrained, trained")
    args = ap.parse_args()

    torch.manual_seed(0)
    seq = TartanAirSequence(REPO_DIR / "data" / "tartan_air" / args.sequence)
    dataset = TACorrespondenceDataset(seq)

    if args.pair is None:
        args.pair = most_textured_pair(seq, dataset)
        print(f"auto-picked pair {args.pair} (most textured)")

    sample = dataset[args.pair]
    batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in sample.items()}

    # SIFT slots in as just another model because it exposes the same interface:
    # (B,3,H,W) -> (B,D,H',W') on the same grid. Same queries, same candidate
    # pool, same metric -- only the descriptor differs.
    builders = {
        "SIFT": lambda: SIFT(step=dataset.sample_step),
        "untrained": lambda: load_model(None),
        "trained": lambda: load_model(args.checkpoint),
    }
    models = {}
    for name in [n.strip() for n in args.models.split(",")]:
        key = {"sift": "SIFT"}.get(name.lower(), name.lower())
        try:
            models[key] = builders[key]()
        except (KeyError, FileNotFoundError) as e:
            print(f"skipping {name!r}: {e}")

    results = {name: analyse(m, batch) for name, m in models.items()}

    img_i, img_j = to_numpy_image(batch["rgb_i"]), to_numpy_image(batch["rgb_j"])

    # default query: whichever correspondence sits nearest the image centre,
    # since edge points are the least interesting and often the most ambiguous
    if args.query is None:
        centre = torch.tensor([img_i.shape[1] / 2, img_i.shape[0] / 2])
        query_n = int(torch.linalg.vector_norm(sample["uv_i"] - centre, dim=-1).argmin())
    else:
        query_n = args.query

    ncols = len(results)
    fig, axes = plt.subplots(3, ncols, figsize=(9 * ncols, 13), squeeze=False)
    for col, (name, res) in enumerate(results.items()):
        draw_match_lines(axes[0][col], img_i, img_j, res, f"[{name}] match lines")
        draw_heatmap(axes[1][col], img_j, res, query_n, f"[{name}] similarity, query #{query_n}")
        draw_pca(axes[2][col], res, f"[{name}] descriptor structure")

    fig.suptitle(f"{args.sequence}  pair {args.pair}  "
                 f"(frames {dataset.pairs[args.pair][0]} -> {dataset.pairs[args.pair][1]})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    for name, res in results.items():
        print(f"\n{name}:")
        for tau in (4, 8, 16):
            print(f"  MMA@{tau:<3} {mma(res['error'], tau):.2%}")
        print(f"  median error {res['error'].median():.1f} px")
        sim = res["similarity"]
        print(f"  peak-minus-mean similarity (higher = sharper): "
              f"{(sim.max(dim=-1).values - sim.mean(dim=-1)).mean():.3f}")


if __name__ == "__main__":
    main()
