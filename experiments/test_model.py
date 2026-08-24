from torch.utils.data import DataLoader

from visual_pose.models.training import test_model
from visual_pose.config import Config
from visual_pose.models.descriptor_cnn import DescriptorCNN
from visual_pose.models.sift import SIFT
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.data_utils.training_dataset import TACorrespondenceDataset
from visual_pose.data_utils.dataset import TartanAirSequence

if __name__ == "__main__":

    seq = TartanAirSequence(REPO_DIR / "data" / "tartan_air" / "P003")
    # model = DescriptorCNN().to(DEVICE)
    model_baseline = SIFT()
    cfg = Config()
    dataset = TACorrespondenceDataset(seq, eval=True, frame_gap=5, **cfg.common())
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False,
                             num_workers=4, pin_memory=(DEVICE.type == "cuda"))

    # the real (untrained) baseline
    test_model(model_baseline, data_loader)

