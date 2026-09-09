from torch.utils.data import DataLoader

from visual_pose.matching import test_model
from visual_pose.config import Config
from visual_pose.models.descriptor_cnn import DescriptorCNN
from visual_pose.models.sift import SIFT
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.data_utils.training_dataset import TACorrespondenceDataset
from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.checkpoints import load_model
from visual_pose.data_utils.loaders import build_wide_baseline_loaders


if __name__ == "__main__":

    seq = TartanAirSequence(REPO_DIR / "data" / "tartan_air" / "P003")
    cfg = Config()
    cfg.wide_val_pairs = 500
    cfg.run_name = "k7_d122_mma70"
    model = load_model(cfg).eval()
    train_loader, val_wide_loader, val_narrow_loader = build_wide_baseline_loaders(cfg)

    # model_baseline = SIFT()
    # cfg = Config()
    # dataset = TACorrespondenceDataset(seq, eval=True, frame_gap=5, **cfg.common())
    # data_loader = DataLoader(dataset, batch_size=8, shuffle=False,
    #                          num_workers=4, pin_memory=(DEVICE.type == "cuda"))

    # the real (untrained) baseline
    test_model(model, val_wide_loader, identity=False)
    # test_model(model_baseline, data_loader, identity=True)


