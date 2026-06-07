Here is a complete, production-ready README.md file designed for your GitHub repository. It includes proper markdown syntax, layout blocks, math equations, and code blocks that perfectly summarize your implementation based on the repository structure and results.

---

markdown
 Custom LSTM Character-Level Text Generation (Tolkien Corpus)

[cite_start]An optimized, from-scratch implementation of a 2-layer stacked Long Short-Term Memory (LSTM) network built in PyTorch without relying on high-level wrappers like nn.LSTM or nn.LSTMCell[cite: 106, 110, 139]. The architecture is designed to map character sequence statistics and autoregressively synthesize text in the distinct literary style of J.R.R. [cite_start]Tolkien[cite: 83, 124, 191].

[cite_start]This project was developed as part of the Sequence Learning (Summer Term 2026) curriculum at the Universität zu Lübeck[cite: 3, 9, 10].

---

 🚀 Key Features

 [cite_start]From-Scratch Layer Design: Pure mathematical execution using elementary linear transformations (nn.Linear) and explicit non-linear activation gates[cite: 110, 135].
 [cite_start]Fused Matrix Efficiency: Combines all four internal gating projections (forget, input, cell candidate, and output) into a single unified matrix multiplication (4  hidden_size) to utilize optimal BLAS execution paths[cite: 136, 142, 143].
 [cite_start]Closed-Loop Autoregression: Employs a dual-phase execution engine using teacher forcing for internal state warm-up, followed by greedy decoding over feedback loops[cite: 190, 191].
 [cite_start]Sequence-First Layout Optimization: Structured tensor layout mapping dimensions directly to (Sequence_Length, Batch_Size, Feature_Dim) to ensure seamless compatibility with cross-entropy tracking[cite: 117, 150, 151].

---

 📁 Repository Structure

text
lstm_from_scratch/
├── models/
│   └── lstm/
│       ├── __init__.py
│       └── lstm.py          <- Core LSTM custom layers & network stack
├── utils/
│   ├── __init__.py
│   ├── configuration.py     <- Attribute dictionary configuration manager
│   ├── datasets.py          <- Shifting & dataset target vector generator
│   └── helper_functions.py  <- Data parsing, hot vector conversion, & saving utilities
├── data/
│   └── chapter1/
│       └── data.pt          <- Serialized preprocessed vocabulary & sample tensors
├── config.json              <- Network hyperparameters & pipeline parameters
├── data_preparation.py      <- Raw text parsing and sequence tokenization script
├── train.py                 <- Training loop engine utilizing Cross-Entropy loss
├── test.py                  <- Autoregressive text generation evaluation script
├── tiny_tolkien.txt         <- Raw training text corpus
└── documentation.txt        <- Thorough project implementation report



---

 ⚙️ Model Architecture

The deep sequence network leverages two recurrent stacked hidden layers and an independent dense linear projection layer:

text
       INPUT: One-Hot Encoded Vector (Alphabet Size: 45)
                       │
                       ▼
       [ Custom LSTM Recurrent Layer 1 ]  (Hidden Size: 128)
                       │
                       ▼
       [ Custom LSTM Recurrent Layer 2 ]  (Hidden Size: 128)
                       │
                       ▼
       [ Fully Connected Output Layer ]   (Dense Projection: 128 ➔ 45)
                       │
                       ▼
      OUTPUT: Predicted Next-Char Logits  (Alphabet Size: 45)



 Recurrent Gate Calculations (Single Time-step)

Inside the custom LSTMCell, execution leverages vectorized parallel operations across a combined hidden weight space prior to element-wise splitting with .chunk():

$$\mathbf{g}_t = \mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b}$$

$$\phi_t = \sigma(\mathbf{g}_{t,\phi}) \quad \iota_t = \sigma(\mathbf{g}_{t,\iota}) \quad \tilde{\mathbf{s}}_t = \tanh(\mathbf{g}_{t,c}) \quad \omega_t = \sigma(\mathbf{g}_{t,\omega})$$

$$\mathbf{s}_t = \phi_t \odot \mathbf{s}_{t-1} + \iota_t \odot \tilde{\mathbf{s}}_t$$

$$\mathbf{h}_t = \omega_t \odot \tanh(\mathbf{s}_t)$$

---

 📈 Training Profile & Convergence Metrics

 Hyperparameter Configurations (config.json)

 
Optimizer: Adam Optimizer 


 Learning Rate ($LR$): $0.001$
 
Total Epochs: $50$ 


 
Batch Size: $1$ (Required to process variable-length sentence strings cleanly) 


 
Vocabulary Size ($d_{one\_hot}$): 45 unique lowercase characters 


 
Trainable Parameters: 226,477 parameters 


 
Hardware Acceleration: CUDA Pipeline (NVIDIA GeForce RTX 4050 Laptop GPU) 



 Optimization Results

The pipeline achieved smooth convergence over 1,450.55 total seconds (~24 minutes):

| Milestone Target | Cross-Entropy Loss Error | Training Progress Sign |
| --- | --- | --- |
| Epoch 01/50 | 3.0155793781 | Initial Baseline (+) 

 |
| Epoch 10/50 | 1.8351824913 | Pattern Structuring (+) 

 |
| Epoch 20/50 | 1.4742580725 | Vocabulary Capture (+) 

 |
| Epoch 30/50 | 1.2046647105 | Phrase Convergence (+) 

 |
| Epoch 40/50 | 0.9727720483 | Syntax Optimization (+) 

 |
| Epoch 50/50 | 0.7748382399 | Final Loss Drop (+) 

 |

---

 📝 Text Generation Performance Analysis

During verification testing (test.py), the model is evaluated by accepting a $30$-character seed initialization sequence to prime hidden states, followed by $400$ closed-loop autonomous steps.

 Inference Output Example

text
Initialization: 
‘will you come with me?’ said ...

Target Ground Truth (Tolkien Reference):
 --- ‘will you come with me?’ said beregond. ‘you may join my mess for this day. i do not know to what company you will be assigned; or the lord may hold you at his own command. but you will be welcome. and it will be well to meet as many men as you may, while there is yet time.’

Custom LSTM Model Closed-Loop Synthesis:
 --- ‘will you come with me?’ said pippin.
 ‘if you have not spoken of these to the great fall. but the shire and flowed upsered, and he was are that is not there. i have wall the strange of the days. and the last that was what with a mestry on the city.’
 said beregond. ‘it is head and stander in the city.’
 said pippin. ‘many bet or so not that the man’s faisht?’
 headsed beregond of the dark hour.
 nged wast papted to the stones



 Behavioral Evaluation

 
Syntactic Mastery: The custom network successfully captures high-level dialogue punctuation framing (‘...’), tokenized turn-taking markers (said pippin., said beregond.), context-appropriate character identities, and structural vocabulary themes native to the source composition (the shire, dark hour, stones, city).


 
Generative Bottlenecks: Operating at a pure character-level means the long-range semantic coherence degrades across large sequence windows, producing minor orthographic non-words (upsered, mestry, faisht).



---

 🛠️ Execution Instructions

 1. Project Dependencies

Ensure you have PyTorch installed within your Python virtual environment:

bash
pip install torch



 2. Prepare the Datasets

Run the preprocessing script to build the sorted alphabet list and generate your tokenized sequence file (data.pt):

bash
python3 data_preparation.py



 3. Run Network Training

Execute the training pipeline. The script optimizes weights across 50 epochs and automatically tracks and preserves the best loss checkpoints under checkpoints/LSTM/LSTM.pt:

bash
python3 train.py



 4. Evaluate and Generate Text

Load your trained network weights to verify performance and evaluate text synthesis from custom seed inputs:

bash
python3 test.py






