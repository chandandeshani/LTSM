# Custom LSTM Character-Level Text Generation

A character-level text generation model implemented from scratch in PyTorch using a custom multi-layer Long Short-Term Memory (LSTM) architecture.

The project trains directly on raw text and learns to generate new text one character at a time using autoregressive generation. Instead of relying on `nn.LSTM`, the recurrent logic is implemented manually using linear layers and explicit gate computations.

The model is trained on a Tolkien text corpus and generates new text with similar punctuation, sentence structure, and writing patterns.

---

## Features

* Custom LSTM implementation without `nn.LSTM`
* Character-level next-token prediction
* Two-layer stacked recurrent architecture
* Efficient gate computation using fused matrix operations
* Teacher forcing during initialization
* Closed-loop autoregressive generation
* Training checkpoint saving
* CUDA GPU support

---

## Project Structure

```text
.
├── models/
│   └── lstm/
│       ├── __init__.py
│       └── lstm.py
│
├── utils/
│   ├── configuration.py
│   ├── datasets.py
│   └── helper_functions.py
│
├── data/
│   └── chapter1/
│       └── data.pt
│
├── checkpoints/
│   └── LSTM/
│
├── config.json
├── data_preparation.py
├── train.py
├── test.py
├── tiny_tolkien.txt
└── documentation.txt
```

---

## Model Architecture

```text
Input (One-Hot Character Vector)
            │
            ▼
Custom LSTM Layer 1
(hidden_size = 128)
            │
            ▼
Custom LSTM Layer 2
(hidden_size = 128)
            │
            ▼
Linear Projection
(128 → 45)
            │
            ▼
Next Character Prediction
```

The network operates at the character level.

Input:

```text
hello
```

Training targets:

```text
h → e
e → l
l → l
l → o
```

---

## Data Preparation

Convert raw text into serialized training data.

```bash
python3 data_preparation.py
```

Output:

```text
data/chapter1/data.pt
```

Generated dataset contains:

* Character vocabulary
* Encoded training samples
* Integer token mapping

---

## Training

Train the model:

```bash
python3 train.py
```

Default configuration:

```json
{
    "epochs": 50,
    "learning_rate": 0.001,
    "hidden_size": 128
}
```

Training uses:

* Optimizer: Adam
* Loss: CrossEntropyLoss
* Device: CUDA (if available)

---

## Training Results

Final training statistics:

| Metric        | Value     |
| ------------- | --------- |
| Epochs        | 50        |
| Parameters    | 226,477   |
| Initial Loss  | 3.015     |
| Final Loss    | 0.775     |
| Training Time | ~1450 sec |

Loss consistently decreased during training:

```text
Epoch 01 → 3.015
Epoch 10 → 1.835
Epoch 20 → 1.474
Epoch 30 → 1.205
Epoch 40 → 0.973
Epoch 50 → 0.775
```

---

## Generate Text

Run inference:

```bash
python3 test.py
```

Generation process:

1. Feed initial prompt (teacher forcing)
2. Initialize hidden states
3. Predict next character
4. Feed prediction back into model
5. Repeat for multiple steps

---

## Example Output

Prompt:

```text
'will you come with me?' said ...
```

Generated:

```text
'will you come with me?' said pippin.

'if you have not spoken of these to the great fall.
but the shire and flowed upsered,
and he was are that is not there...
```

Observed behavior:

* Produces realistic punctuation
* Learns dialogue formatting
* Reuses names and recurring patterns
* Maintains short-term coherence
* Occasionally generates non-words

---

## Configuration

Edit `config.json`:

```json
{
    "training": {
        "epochs": 50,
        "learning_rate": 0.001
    },

    "testing": {
        "prompt_length": 30,
        "closed_loop_steps": 400
    },

    "model": {
        "d_one_hot": 45,
        "d_lstm": 128
    }
}
```

---

## Notes

This project focuses on understanding sequence modeling and implementing recurrent neural networks manually rather than achieving state-of-the-art language generation quality.

The implementation demonstrates:

* recurrent state propagation
* sequence learning
* autoregressive decoding
* custom deep learning architecture design
