"""
Stage-1 sensitivity pass: change ONE knob at a time from a baseline and see
whether val MMA actually moves. Knobs that don't move get frozen and never
tuned again -- that is what makes a 20-parameter space tractable.

    .venv/bin/python -m experiments.sweep

    ... -m experiments.sweep --report        # just print the table, run nothing

Runs on a deliberately cheap PROXY (2 sequences, single frame_gap, few epochs)
so each trial is minutes not hours. The bet is that the *ranking* of configs
survives the shrink even though absolute numbers don't.

That bet fails for regularizers -- weight decay, augmentation strength, dropout
cost you early and pay late, so a short proxy will always tell you to turn them
down. Those are deliberately NOT in VARIATIONS below; set them from priors and
only ever verify them on a full-length run.

Pruning is off here on purpose: a sensitivity table needs every trial's final
number to be comparable. Turn on MedianPruner for stage 2 (TPE refinement of
whichever knobs survive this pass), where the point is to save time rather than
to compare fairly.
"""
import argparse
import sys
from dataclasses import replace

import optuna
from optuna.trial import create_trial
import torch
from torch.utils.data import DataLoader, Subset

from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.models.descriptor_cnn import DescriptorCNN
from visual_pose.training import (train_val_model, set_up_loss_optimizer_lr_scheduler,
                                         test_model, mma)

# Cheap stand-in for the real problem.
# P000 (bright indoor hospital) + P006 (dark japanese alley) keeps the
# bright/dark contrast that makes the norm and jitter knobs meaningful, while
# avoiding P002/P005: those are outdoor, so most of the frame is sky at depth
# 65504 and ~38% of their pairs fall below num_correspondences. Every one of
# those triggers a retry, and retries dominated the per-item cost.
PROXY = dict(
    train_sequences=("P000", "P006"),
    frame_gap=(5,),
    num_epochs=6,
    print_freq=200,
)

# Correspondences within a pair are highly correlated (same image), so the
# effective sample size tracks the PAIR count, not the 512-per-pair count.
# Per-pair MMA spread is ~10pp, so SE ~ 10/sqrt(n_pairs): 607 pairs buys 0.4pp,
# 150 buys 0.8pp. Both sit under the run-to-run training noise we are measuring
# against, so the extra 457 pairs are 4x the eval cost for nothing.
VAL_PAIRS = 150

# A sensitivity pass only needs the best value, not the curve.
EVAL_EVERY = 2

# These change the compute per step, not just the weights, so they run several
# times slower than the rest. Ordered last so the bulk of the table lands early
# and the run can be stopped once it is conclusive.
#   body_strides=(1,2,2) -> stride 4, a 120x160 map: 19200 eval candidates
#                           instead of 4800, and conv1 at full resolution
#   large kernels        -> 7x7 in layer 1 is ~5x the compute of 3x3
#   body_channels up     -> wider everywhere
EXPENSIVE = ("body_strides", "body_kernel_sizes", "body_channels", "sample_step")

# Repeat the baseline at several seeds. This is the most important entry here:
# without a noise floor, "config X beat baseline by 1pp" is uninterpretable.
# The report uses the spread across these as the significance threshold.
SEEDS = [0, 1, 2]

# One entry per knob, listing values to try. Any value equal to the baseline is
# skipped automatically, so listing the default is harmless.
VARIATIONS = {
    # -- loss / optimisation: these show up within a couple of epochs
    "temperature":    [0.02, 0.07, 0.20],
    "learning_rate":  [0.003, 0.01, 0.03],
    "optimizer":      ["sgd", "adamw"],
    # batch_size is coupled to learning_rate (roughly linear scaling), so a
    # solo result here is only suggestive -- the pair needs joint tuning
    "batch_size":     [4, 8, 16],

    # -- architecture
    "norm":              ["batch", "group"],
    "descriptor_dim":    [64, 128, 256],
    "body_channels":     [(32, 64, 128), (64, 128, 256), (96, 192, 384)],
    # kernel 5 takes the receptive field from 15px to 29px, at 2.8x the compute
    # per layer; the mixed one is wide-early/narrow-late
    "body_kernel_sizes": [(3, 3, 3), (5, 5, 5), (7, 5, 3)],
    # NOTE (1,2,2) is total stride 4: a 120x160 feature map, so 19200 eval
    # candidates instead of 4800. Finer localisation but a 4x harder retrieval
    # task and much slower -- expect this trial to take several times longer.
    "body_strides":      [(2, 2, 2), (1, 2, 2)],

    # -- data / correspondence generation
    # (2,5,10) is deliberately absent: it triples the pair count, so at fixed
    # epochs it also triples the gradient steps -- the trial would confound
    # "more baseline diversity" with "3x more training". (10,) keeps the pair
    # count roughly equal, so it cleanly tests harder pairs.
    "frame_gap":           [(5,), (10,)],
    "num_correspondences": [256, 512, 1024],
    "sample_step":         [4, 8, 12],
    "darkness_threshold":  [0.0, 0.05, 0.15],
}

# Measured as inert or uninterpretable on this proxy, so running them would burn
# ~10 min each to learn nothing. Kept here as a record of what was ruled out and
# why, and where each should actually be tested.
SKIPPED = {
    # changes 0.02-0.18% of correspondences on P000/P006 -- both are enclosed,
    # so there is no distant geometry for a higher ceiling to admit
    "max_depth": "inert on the proxy; only matters on outdoor sequences",
    # changes 1.6-2.1%; already frozen in Config after direct measurement
    "occlusion_tol": "inert; 2 of 4796 correspondences at the extremes",
    # PROXY-HOSTILE: regularizers cost accuracy early and pay late, so a 6-epoch
    # run always says turn them down. We would not act on the result either way,
    # so measuring it is pure cost. Test these with full-length A/B runs.
    "jitter": "proxy-hostile -- needs a full-length A/B",
    "weight_decay": "proxy-hostile -- needs a full-length A/B",
}

# Deliberately not swept: num_epochs/num_workers/print_freq (not model quality),
# run_name/seed (bookkeeping), val_sequence/val_frame_gap/checkpoint_tau (they
# define the metric -- changing them makes runs incomparable), momentum and
# min_lr_factor (literature defaults, and momentum is coupled to learning_rate).

# ---------------------------------------------------------------------------
# Stage 2: receptive field.
#
# Stage 1 said RF dominates everything else by an order of magnitude. Its four
# architecture variants line up monotonically on RF alone:
#     RF  9 -> 15.89% | RF 15 -> 17.86% | RF 23 -> 27.26% | RF 29 -> 36.46%
# The trend had not turned over at 29, so the job now is to find where it does.
#
# The baseline moves to stage 1's two winners, kernel (5,5,5) and temperature
# 0.02. Combining winners is itself untested, so temperature=0.07 is included to
# check that 0.02 still helps at the larger RF rather than assuming it carries.
#
# Kernel 9 costs 3.4M params against 0.4M at baseline -- quadratic in k. Dilation
# reaches the same distances at kernel-3 cost and is the better lever, deferred
# until it is understood rather than added blind.
# ---------------------------------------------------------------------------
STAGE2_PROXY = dict(
    train_sequences=("P000", "P006"),
    frame_gap=(5,),
    num_epochs=6,
    print_freq=200,
    body_kernel_sizes=(5, 5, 5),
    temperature=0.02,
)

STAGE2_VARIATIONS = {
    # RF 29 (baseline) -> 43 -> 57. Where does it stop paying?
    "body_kernel_sizes": [(5, 5, 5), (7, 7, 7), (9, 9, 9)],
    # sample_step is COUPLED to RF: at RF 15, step 8 beat both 4 and 12 by ~4pp,
    # plausibly because adjacent samples then barely overlap. At RF 29 the
    # matched spacing should be wider, so 8 may now be too dense -- overlapping
    # receptive fields make InfoNCE push apart descriptors that should be alike.
    "sample_step": [8, 12, 16],
    # does the stage-1 temperature win survive the new baseline?
    "temperature": [0.02, 0.07],
    # both were LIVE in stage 1 (+0.97, +0.89); re-check at the new baseline
    "batch_size": [4, 8],
    "frame_gap": [(5,), (10,)],
}

STAGE2_EXPENSIVE = ("body_kernel_sizes", "sample_step")

# ---------------------------------------------------------------------------
# Stage 3: receptive field again, but at constant cost.
#
# Stage 2 confirmed RF is the whole story and had not saturated:
#     RF 15 -> 17.86% | 29 -> 36.41% | 43 -> 48.13% | 57 -> 56.36%
# Gains per RF pixel are decelerating (1.5, 0.84, 0.59pp) but still positive.
#
# Kernel 9 got RF 57 for 3.4M params -- a full 6-environment run at that size is
# 20-40h. Dilation reaches further for 0.4M, so stage 3 goes back to kernel 3 and
# varies dilation instead. Parameters, FLOPs and layer count are all held fixed,
# so any change in the metric can only be receptive field.
#
# Dilations grow with depth on purpose. A dilated layer only samples a sublattice
# of its input, so stacking the SAME dilation leaves permanent blind spots (the
# gridding artifact) -- (3,3,3) at d=3 throughout would see 1/9 of its own
# receptive field. Rising dilations let each layer's gaps be covered by the
# denser layer beneath it.
#
# (1,2,4) is the conservative option: RF 43 with tight coverage, matching kernel
# 7's reach at a quarter of its parameters. (1,3,9) reaches RF 87 but spaces the
# third layer's taps 36 input pixels apart, so coverage is thinner -- if it
# underperforms (1,2,4) despite double the RF, gridding is the likely reason.
# ---------------------------------------------------------------------------
STAGE3_PROXY = dict(
    train_sequences=("P000", "P006"),
    frame_gap=(5,),
    num_epochs=6,
    print_freq=200,
    body_kernel_sizes=(3, 3, 3),   # back to cheap kernels; dilation does the work
    body_dilations=(1, 2, 4),      # RF 43, the conservative choice, as baseline
    # matches stage 2's baseline, NOT the Config default. Stage 2's kernel-7
    # trial also reached RF 43 but at 100% coverage instead of 39%, so it is the
    # controlled comparison for whether coverage matters -- and that only works
    # if temperature is held equal across the two stages.
    temperature=0.02,
)

# Measured RF / params / coverage for every option below, so the sweep spans the
# plane deliberately rather than by accident. "Coverage" is the fraction of
# pixels inside the RF box that actually reach the output -- measured by
# backprop from one output cell -- and it is what dilation trades away.
#
#   config                    RF   params  coverage
#   k3 d(1,1,1)               15    405K     100%
#   k3 d(1,2,2)               27    405K    60.5%
#   k3 d(1,2,4)  <- baseline  43    405K    39.4%
#   k3 d(1,3,9)               87    405K     9.6%
#   k5 d(1,2,2)               53   1.06M     100%
#   k5 d(1,2,4)               85   1.06M     100%
#   k9 d(1,1,1)               57   3.37M     100%
#   4L dense top d(1,2,4,1)   59    995K    58.2%
#   4L all dense              31    995K     100%
#
# Note k5+dilation keeps FULL coverage: five taps are dense enough relative to
# their span that the gaps close, where three taps are not. k5 d(1,2,4) is
# therefore strictly better than kernel 9 on all three axes at once.
STAGE3_VARIATIONS = {
    "body_dilations": [(1, 1, 1), (1, 2, 2), (1, 2, 4), (1, 3, 9)],
    # at RF 85 a descriptor summarises a far larger patch than at RF 15, so 128
    # dims may bind now where they were flat before
    "descriptor_dim": [128, 256],
    # stage 2's other two survivors were ~1pp against a 0.63pp floor, so both
    # are marginal; re-check once at the new baseline
    "sample_step": [8, 16],
    "frame_gap": [(5,), (10,)],
}

# Trials that need several fields moved together, which one-knob-at-a-time
# cannot express. Each entry is a full set of overrides.
STAGE3_COMPOUND = {
    # RF 53 and 85 at FULL coverage -- the pair that isolates reach from density,
    # since coverage is pinned at 100% and only RF moves
    "k5_d122_rf53_cov100": dict(body_kernel_sizes=(5, 5, 5), body_dilations=(1, 2, 2)),
    "k5_d124_rf85_cov100": dict(body_kernel_sizes=(5, 5, 5), body_dilations=(1, 2, 4)),
    # the expensive 100%-coverage control: RF 57 for 3.37M params
    "k9_rf57_cov100": dict(body_kernel_sizes=(9, 9, 9), body_dilations=(1, 1, 1)),
    # k5/k9 with dilation all hold 100% coverage, so these four together give a
    # saturation curve at CONSTANT density: RF 53, 85, 105, 169. That is the
    # cleanest read on where reach stops paying.
    "k9_d122_rf105_cov100": dict(body_kernel_sizes=(9, 9, 9), body_dilations=(1, 2, 2)),
    "k9_d124_rf169_cov100": dict(body_kernel_sizes=(9, 9, 9), body_dilations=(1, 2, 4)),
    # RF 173 at 52% coverage for 1.06M -- pairs with k9_d124 (RF 169 at 100% for
    # 3.37M) to test coverage at the FAR end of the range, the way the k3 d(1,2,4)
    # baseline pairs with stage 2's kernel 7 at RF 43.
    "k5_d139_rf173_cov52": dict(body_kernel_sizes=(5, 5, 5), body_dilations=(1, 3, 9)),
    # NOTE RF 169 spans 35% of the image height. Too LARGE a receptive field is
    # its own failure mode -- the descriptor starts encoding scene layout that
    # moves with the camera rather than the local surface -- so this range should
    # bracket the optimum rather than just approach it from below.
    # a dense 4th layer on top of dilated ones: RF 43 -> 59 AND coverage
    # 39.4% -> 58.2%, i.e. it repairs the gridding it sits above
    "4L_densetop_rf59_cov58": dict(
        body_channels=(64, 128, 256, 256), body_kernel_sizes=(3, 3, 3, 3),
        body_strides=(2, 2, 2, 1), body_dilations=(1, 2, 4, 1)),
    # depth WITHOUT extra reach-per-layer: RF 31 at 100% coverage. Paired with
    # k3 d(1,2,2) (RF 27, 60% coverage, 405K) it separates "more layers" from
    # "more reach" at nearly equal RF.
    "4L_alldense_rf31_cov100": dict(
        body_channels=(64, 128, 256, 256), body_kernel_sizes=(3, 3, 3, 3),
        body_strides=(2, 2, 2, 1), body_dilations=(1, 1, 1, 1)),
}

# dilation is free; anything raising kernel size or layer count is not
STAGE3_EXPENSIVE = ("k5_", "k9_", "4L_")

# ---------------------------------------------------------------------------
# Stage 4: capacity vs reach, at 100% coverage throughout.
#
# Stage 3 settled two things. Coverage matters about as much as reach -- RF 85 at
# 100% scored 48.65% while RF 87 at 9.6% scored 28.42%, same reach, +20pp for
# density -- so every config here keeps full coverage. And the RF optimum scales
# with capacity: k5 (1.06M) peaked below RF 85, k9 (3.37M) peaked near RF 105.
#
# Baseline is stage 3's winner, k9 d(1,2,2), RF 105 at 58.87%. Deliberately a
# STRONG baseline: stage 3 used a weak one (k3 d(1,2,4)) and its seed spread blew
# out to 2.60% from the usual 0.6%, which buried every sub-2pp effect.
#
# Two comparisons carry most of the value here, both holding one axis fixed:
#   k7 d(1,2,3)      RF 103 at 0.61x params -- same reach, less capacity
#   wide k9 d(1,2,2) RF 105 at 2.24x params -- same reach, more capacity
# Together they say whether anything beyond receptive field is binding at all.
# ---------------------------------------------------------------------------
STAGE4_PROXY = dict(
    train_sequences=("P000", "P006"),
    frame_gap=(5,),
    num_epochs=6,
    print_freq=200,
    body_kernel_sizes=(9, 9, 9),
    body_dilations=(1, 2, 2),
    temperature=0.02,
)

STAGE4_VARIATIONS = {
    # RF 89 / 105 / 137 at fixed 3.37M -- refine the peak found in stage 3
    "body_dilations": [(1, 1, 2), (1, 2, 2), (1, 2, 3)],
    # both were blunted by stage 3's inflated noise floor (+1.72 and +1.12
    # against a 2.60% spread), and descriptor_dim is more plausible now that a
    # descriptor summarises a 105px patch rather than a 15px one
    "sample_step": [8, 16],
    "descriptor_dim": [128, 256],
}

STAGE4_COMPOUND = {
    # -- cheaper than baseline: is k9 overkill? --
    "k7_d122_rf79_x0.61": dict(body_kernel_sizes=(7, 7, 7), body_dilations=(1, 2, 2)),
    # RF 103 ~= baseline's 105 at 61% of the parameters. If this matches, reach is
    # all that matters and the extra capacity is wasted.
    "k7_d123_rf103_x0.61": dict(body_kernel_sizes=(7, 7, 7), body_dilations=(1, 2, 3)),

    # -- same reach, more capacity: is capacity binding at RF 105? --
    # stage 1 found width flat at RF 15; this re-asks it where reach is no longer
    # the bottleneck
    "wide_k9_d122_rf105_x2.24": dict(body_channels=(96, 192, 384),
                                     body_kernel_sizes=(9, 9, 9),
                                     body_dilations=(1, 2, 2)),

    # -- bigger models, further reach: does the peak keep moving out? --
    "k11_d122_rf131_x1.49": dict(body_kernel_sizes=(11, 11, 11), body_dilations=(1, 2, 2)),
    "k13_d122_rf157_x2.08": dict(body_kernel_sizes=(13, 13, 13), body_dilations=(1, 2, 2)),

    # depth with a cheap dense top: RF 121 for only 1.18x, since the 4th layer is
    # 3x3. Stage 3's depth trials were confounded by coverage; this one is not.
    "4L_k9top3_rf121_x1.18": dict(body_channels=(64, 128, 256, 256),
                                  body_kernel_sizes=(9, 9, 9, 3),
                                  body_strides=(2, 2, 2, 1),
                                  body_dilations=(1, 2, 2, 1)),
}

# ordered by parameter count relative to the 3.37M baseline, so the cheap
# answers land first and the 5-8.7M trials run last
STAGE4_EXPENSIVE = ("k11_", "k13_", "wide_", "4L_")

STAGES = {
    "sensitivity":     dict(proxy=PROXY, variations=VARIATIONS, expensive=EXPENSIVE),
    "receptive_field": dict(proxy=STAGE2_PROXY, variations=STAGE2_VARIATIONS,
                            expensive=STAGE2_EXPENSIVE),
    "dilation":        dict(proxy=STAGE3_PROXY, variations=STAGE3_VARIATIONS,
                            compound=STAGE3_COMPOUND, expensive=STAGE3_EXPENSIVE),
    "capacity":        dict(proxy=STAGE4_PROXY, variations=STAGE4_VARIATIONS,
                            compound=STAGE4_COMPOUND, expensive=STAGE4_EXPENSIVE),
}

STORAGE = f"sqlite:///{REPO_DIR / 'checkpoints' / 'sweep.db'}"


def trial_configs(stage: str = "sensitivity"):
    """Baseline seeds first (to get a noise floor), then one knob at a time."""
    spec = STAGES[stage]
    base = Config(**spec["proxy"])
    expensive = spec["expensive"]

    def is_costly(label):
        return any(label.startswith(e) for e in expensive)

    cheap, costly = [], []
    for knob, values in spec["variations"].items():
        for v in values:
            if v == getattr(base, knob):
                continue          # that IS the baseline, no need to repeat it
            label = f"{knob}={v}"
            (costly if is_costly(label) else cheap).append((label, {knob: v}))

    # compound trials move several fields at once, which one-knob-at-a-time
    # cannot express (a 4-layer net needs channels, kernels, strides and
    # dilations changed together or the length asserts fire)
    for label, overrides in spec.get("compound", {}).items():
        (costly if is_costly(label) else cheap).append((label, dict(overrides)))

    seeds = [(f"baseline.seed{s}", {"seed": s}) for s in SEEDS]
    return base, seeds + cheap + costly


# Fields that change what the loaders produce. Trials varying only model or
# optimiser settings can reuse loaders outright -- worth doing because
# persistent_workers means each rebuild spawns 8 processes, and on macOS spawn
# re-imports torch in every one. About half the trials here hit this cache.
_DATA_FIELDS = ("train_sequences", "val_sequence", "frame_gap", "val_frame_gap",
                "num_correspondences", "sample_step", "darkness_threshold",
                "max_depth", "occlusion_tol", "jitter", "batch_size",
                "num_workers", "seed")
_LOADER_CACHE = {}


def get_loaders(cfg: Config):
    key = tuple(str(getattr(cfg, f)) for f in _DATA_FIELDS)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]

    train_loader, val_loader = build_loaders(cfg)
    # strided subset so the reduced val set still spans the whole sequence --
    # a contiguous block would sample one part of the trajectory
    n = len(val_loader.dataset)
    if VAL_PAIRS < n:
        idx = list(range(0, n, n // VAL_PAIRS))[:VAL_PAIRS]
        val_loader = DataLoader(Subset(val_loader.dataset, idx),
                                batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers,
                                persistent_workers=cfg.num_workers > 0)

    _LOADER_CACHE.clear()   # keep only one live set; workers are not free
    _LOADER_CACHE[key] = (train_loader, val_loader)
    return train_loader, val_loader


def run_one(base: Config, overrides: dict, label: str, trial=None) -> float:
    cfg = replace(base, run_name=f"sweep/{label}", **overrides)
    torch.manual_seed(cfg.seed)

    train_loader, val_loader = get_loaders(cfg)

    model = DescriptorCNN(
        body_channels=list(cfg.body_channels),
        body_kernel_sizes=list(cfg.body_kernel_sizes),
        body_strides=list(cfg.body_strides),
        body_dilations=list(cfg.body_dilations),
        descriptor_dim=cfg.descriptor_dim,
        norm=cfg.norm,
    ).to(DEVICE)

    loss_fn, optimizer, lr_scheduler = set_up_loss_optimizer_lr_scheduler(
        model=model, learning_rate=cfg.learning_rate, momentum=cfg.momentum,
        num_epochs=cfg.num_epochs, weight_decay=cfg.weight_decay,
        min_lr_factor=cfg.min_lr_factor, optimizer=cfg.optimizer,
        temperature=cfg.temperature)

    best = [0.0]

    def on_epoch_end(epoch, val_mma):
        best[0] = max(best[0], val_mma)
        if trial is not None:
            trial.report(val_mma, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    train_val_model(model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler,
                    num_epochs=cfg.num_epochs, print_freq=cfg.print_freq,
                    checkpoint_dir=cfg.checkpoint_dir, checkpoint_tau=cfg.checkpoint_tau,
                    on_epoch_end=on_epoch_end, eval_every=EVAL_EVERY)
    return best[0]


def report(study, base):
    rows = [(t.user_attrs.get("label", "?"), t.value)
            for t in study.trials if t.value is not None]
    if not rows:
        print("no completed trials yet")
        return

    seeds = [v for l, v in rows if l.startswith("baseline.seed")]
    if not seeds:
        print("no baseline seeds completed -- cannot judge significance yet")
        return
    mean = sum(seeds) / len(seeds)
    # full spread across seeds, not std: with 3 samples the spread is the more
    # honest statement of "differences this small are indistinguishable"
    noise = (max(seeds) - min(seeds)) if len(seeds) > 1 else 0.0

    print(f"\nbaseline: {mean:.2%} mean over {len(seeds)} seeds, "
          f"spread {noise:.2%}  <-- significance threshold")
    print(f"\n{'config':26} {'val MMA@8':>10} {'vs baseline':>12}   verdict")
    for label, v in sorted(rows, key=lambda r: -r[1]):
        if label.startswith("baseline.seed"):
            continue
        d = v - mean
        verdict = "LIVE" if abs(d) > noise else "flat -- freeze it"
        print(f"{label:26} {v:>10.2%} {d:>+12.2%}   {verdict}")
    print("\nOnly refine the LIVE knobs. Treat jitter/weight_decay verdicts as "
          "unreliable here -- short runs always favour less regularization.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print results, run nothing")
    ap.add_argument("--stage", default="capacity", choices=list(STAGES),
                    help="which sweep to run; each keeps its own study")
    args = ap.parse_args()

    base, configs = trial_configs(args.stage)
    (REPO_DIR / "checkpoints").mkdir(exist_ok=True)

    # No sampler is involved: this pass is a designed comparison over an
    # explicit list, so each finished trial is *recorded* via add_trial rather
    # than requested via ask/tell. Going through a sampler here is what made
    # BruteForceSampler call study.stop() -- it saw no suggest_* calls, decided
    # the space was exhausted, and that is illegal under ask/tell.
    # Stage 2 (TPE refinement of the LIVE knobs) is where a sampler belongs.
    # separate study per stage: they have different baselines, so mixing
    # them would make the noise floor and every delta meaningless
    study = optuna.create_study(study_name=args.stage, storage=STORAGE,
                                direction="maximize", load_if_exists=True)

    if args.report:
        report(study, base)
        return

    # No params and no distributions. An OFAT pass has no search space, and a
    # CategoricalDistribution over the config list is actively harmful: Optuna
    # pins a distribution on first use, so editing VARIATIONS later makes every
    # new trial incompatible with the stored study. The label lives in
    # user_attrs, which has no such constraint.
    done = {t.user_attrs.get("label") for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE}
    todo = [(l, o) for l, o in configs if l not in done]
    print(f"stage {args.stage!r}: {len(configs)} configs, {len(done)} already done, {len(todo)} to run\n")

    for n, (label, overrides) in enumerate(todo, 1):
        print(f"=== [{n}/{len(todo)}] {label} ===")
        try:
            value = run_one(base, overrides, label.replace("=", "_"), trial=None)
        except KeyboardInterrupt:
            print("interrupted -- completed trials are in the study, rerun to continue")
            sys.exit(1)
        study.add_trial(create_trial(
            params={}, distributions={}, value=value,
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs={"label": label, "overrides": str(overrides)},
        ))
        print(f"--> {label}: {value:.2%}\n")

    report(study, base)


if __name__ == "__main__":
    main()
