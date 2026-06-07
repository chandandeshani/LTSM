import torch
import torch.nn as nn
import os
import time
from pathlib import Path
from models.lstm.lstm import LSTM
from utils.configuration import Configuration
from utils import helper_functions as helpers

def train():

    # Load the user configurations
    cfg = Configuration(str(Path(__file__).resolve().parent / "config.json"))

    # Print some information to console
    print("Model name:", cfg.model.name)

    # Set device on GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)  # ADD THIS LINE


    # TODO: initialize the model here
    #       model = LSTM(...

    dataset, dataloader = helpers.build_dataloader(
        cfg=cfg,
        batch_size=1
    )

    model = LSTM(
        input_size=dataset.d_one_hot,
        hidden_size=128,
        output_size=dataset.d_one_hot
    ).to(device)


    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt file
    if cfg.training.continue_training:
        print('Restoring model (that is the network\'s weights) from file...')
        model.load_state_dict(
            torch.load(
                os.path.join(
                    str(Path(__file__).resolve().parent),
                    "checkpoints",
                    cfg.model.name + ".pt"
                )
            )
        )
        model.train()

    # TODO: set up the optimizer and loss

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()


    #
    # Set up a list to save and store the epoch errors
    epoch_errors = []
    best_error = float("inf")

    #
    # Set up the training dataloader
    dataset, dataloader = helpers.build_dataloader(
        cfg=cfg,
        batch_size=1
    )

    """
    TRAINING
    """
    a = time.time()

    #
    # TODO: implement the main training loop here

    # Start the training and iterate over all epochs
    for epoch in range(cfg.training.epochs):

        epoch_start_time = time.time()

        # List to store the errors for each sequence
        sequence_errors = []

        #
        # TODO: implement the main training loop here
        #       for ...

        for batch_idx, (inputs, targets) in enumerate(dataloader):

            model.train()

            inputs = inputs.to(device).transpose(0, 1)
            targets = targets.to(device).transpose(0, 1)

            h, s = model.initial_states
            h = h.to(device)
            s = s.to(device)

            optimizer.zero_grad()

            outputs, (h, s) = model(
                inputs,
                (h, s)
            )

            outputs = outputs.reshape(
                -1,
                outputs.shape[-1]
            )

            targets = targets.reshape(-1)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            # The following line saves the batch error and should be inside the training loop
            # So please indent the following line accordingly
            sequence_errors.append(loss.item())

        epoch_errors.append(
            sum(sequence_errors) / len(sequence_errors)
        )

        # Create a plus or minus sign for the training error
        train_sign = "(-)"

        if epoch_errors[-1] < best_error:

            train_sign = "(+)"

            best_error = epoch_errors[-1]

            # Save the model to file (if desired)
            if cfg.training.save_model:

                helpers.save_model_to_file(
                    model_src_path=str(Path(__file__).resolve().parent),
                    cfg=cfg,
                    epoch=epoch,
                    best_error=best_error,
                    model=model
                )

        # Print progress to the console
        print(
            f"Epoch {str(epoch+1).zfill(len(str(cfg.training.epochs)))}"
            f"/{str(cfg.training.epochs)} took "
            f"{str(round(time.time() - epoch_start_time, 2)).ljust(5, '0')} "
            f"seconds. \t\tAverage epoch training error: "
            f"{train_sign}"
            f"{str(round(epoch_errors[-1], 10)).ljust(12, ' ')}"
        )

    b = time.time()

    print(
        '\nTraining took '
        + str(round(b - a, 2))
        + ' seconds.\n\n'
    )


if __name__ == "__main__":
    train()

    print("Done.")
