import torch
import os
from utils import datasets

def build_dataloader(cfg, batch_size):
    """
    This function creates a dataset and return the appropriate dataloader to
    iterate over this dataset
    :param cfg: The general configurations of the model
    :param batch_size: The number of samples per batch
    :return: A PyTorch dataloader object
    """

    # Set up a dataset and dataloader
    dataset = datasets.TolkienDataset(
        dataset_name=cfg.dataset.name
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2
    )

    return dataset, dataloader


def save_model_to_file(model_src_path, cfg, epoch, best_error, model):

    model_save_path = os.path.join(
        model_src_path, "checkpoints", cfg.model.name
    )

    os.makedirs(model_save_path, exist_ok=True)

    # Save model weights to file
    torch.save(model.state_dict(),
            os.path.join(model_save_path, cfg.model.name + ".pt"))

    # Copy the configurations and add a results entry
    cfg["results"] = {
        "current_epoch": epoch + 1,
        "best_train_error": best_error,
    }

    # Save the configuration and current performance to file
    cfg.save(model_save_path)


def one_hot_to_char(one_hot_vector, alphabet):
    """
    Converts a one-hot vector into a character given an alphabet.
    """
    idx = int(torch.argmax(torch.as_tensor(one_hot_vector)).item())
    return alphabet[idx]


def softmax_to_one_hot(soft):
    """
    Converts a softmax output into a discretized one-hot vector, where the
    largest value of the softmax is encoded as one.
    :param soft: The softmax vector
    :return: The discretized one-hot torch tensor (same device as `soft`)
    """
    one_hot_vector = torch.zeros_like(soft, dtype=torch.int8)
    one_hot_vector[torch.argmax(soft)] = 1
    return one_hot_vector
