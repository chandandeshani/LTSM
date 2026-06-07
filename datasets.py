import torch
import torch.nn.functional as F
from pathlib import Path

class TolkienDataset(torch.utils.data.Dataset):
    """
    Returns (input_seq, target_seq) for each line.
    target_seq is input_seq shifted by one timestep (next-char prediction).
    """

    def __init__(self, dataset_name):
        data_path = (
            Path(__file__).resolve().parent.parent
            / "data" / dataset_name / "data.pt"
        )
        bundle = torch.load(data_path, weights_only=False)
        self.alphabet = bundle["alphabet"]
        self.samples = bundle["samples"]
        self.d_one_hot = len(self.alphabet)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # Input is one-hot (model interface); target is indices (for CE loss).
        input_seq = F.one_hot(sample[:-1], num_classes=self.d_one_hot).float()
        target_seq = sample[1:]
        return input_seq, target_seq
