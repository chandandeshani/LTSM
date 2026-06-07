import torch
import torch.nn as nn
import os
from pathlib import Path
from models.lstm.lstm import LSTM
from utils.configuration import Configuration
from utils import helper_functions as helpers

def test():

    # Load the user configurations
    cfg = Configuration(str(Path(__file__).resolve().parent / "config.json"))

    # Print some information to console
    print("Model name:", cfg.model.name)

    # Set device on GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: initialize the model here
    #       model = LSTM(...
    dataset, dataloader = helpers.build_dataloader(
        cfg=cfg, batch_size=1
    )

    model = LSTM(
        input_size=dataset.d_one_hot,
        hidden_size=cfg.model.d_lstm,
        output_size=dataset.d_one_hot
    ).to(device)

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    # Load the trained weights from the checkpoints into the model
    model.load_state_dict(torch.load(os.path.join(str(Path(__file__).resolve().parent),
                                              "checkpoints",
                                               cfg.model.name,
                                               cfg.model.name+".pt"),
                                  map_location=device))
    model.eval()

    #
    # Set up the dataloader for the testing
    dataset, dataloader = helpers.build_dataloader(
        cfg=cfg, batch_size=1
    )

    """
    TESTING
    """

    prompt_length = cfg.testing.prompt_length
    cl_steps = cfg.testing.closed_loop_steps

    # Iterate over the batches
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        h, s = model.initial_states
        h = h.to(device)
        s = s.to(device)

        # Move data to the desired device and transpose
        inputs = inputs.to(device=device).transpose(0, 1)
        targets = targets.to(device=device).transpose(0, 1)

        # Start with the first teacher forcing characters and continue
        x = inputs[:prompt_length]
        y_hat, (h, s) = model(x, (h, s))
        y_hat = nn.functional.softmax(y_hat, dim=-1)
        y_hat_last = helpers.softmax_to_one_hot(soft=y_hat[-1, 0]).unsqueeze(0).unsqueeze(0)

        # Append the model output to the input
        x = torch.cat((x, y_hat_last), dim=0)

        for t in range(cl_steps):

            # Generate a prediction and apply softmax
            y_hat, (h, s) = model(y_hat_last.float(), (h, s))
            y_hat = nn.functional.softmax(y_hat, dim=-1)

            # Convert the last softmax output to a one-hot vector
            y_hat_last = helpers.softmax_to_one_hot(soft=y_hat[-1, 0]).unsqueeze(0).unsqueeze(0)

            # Append the model output to the input and continue
            x = torch.cat((x, y_hat_last), dim=0)

        inputs = inputs.detach().cpu()
        tolkien_text = []

        for t in range(len(inputs[:prompt_length + cl_steps])):
            y_t = helpers.one_hot_to_char(
                one_hot_vector=inputs[t, 0],
                alphabet=dataset.alphabet
            )
            tolkien_text.append(y_t[0])

        model_text = []
        for t in range(len(x)):
            x_t = helpers.one_hot_to_char(
                one_hot_vector=helpers.softmax_to_one_hot(soft=x[t, 0]),
                alphabet=dataset.alphabet
            )
            model_text.append(x_t[0])

        tolkien_text = "".join(tolkien_text)
        model_text = "".join(model_text)

        print(f"Initialization: {tolkien_text[:prompt_length]}...")
        print(f"\nTolkien:\n --- {tolkien_text}")
        print(f"\nLSTM model:\n --- {model_text}")

        exit()

if __name__ == "__main__":
    test()

    print("Done.")
