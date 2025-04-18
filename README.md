🧠 Predicting Chess Game Outcomes Using Deep Learning (MLP)
📌 Overview
This project investigates the use of deep learning models, particularly Multilayer Perceptrons (MLPs), to predict the outcome of a chess position—win, draw, or loss—based purely on the board state. It involves parsing a dataset of millions of chess games into FEN strings, converting them into numerical feature vectors, and training a neural network on this representation.

![image](https://github.com/user-attachments/assets/140d9918-f429-46a7-824b-afe167d087eb)


🧰 Technologies & Frameworks Used
Python 3.x

TensorFlow (with Keras API) – core deep learning framework

Intel oneDNN – for CPU-accelerated deep learning operations

NumPy, Pandas – data processing and transformation

Matplotlib – visualizing training performance

Stockfish – chess engine used for comparative evaluation

🧠 Neural Network Architecture
🔷 Initial Model
Input Layer: 73-dimension feature vector (encoded FEN)

Hidden Layers: 2 layers with 64 neurons

Activation: ReLU

Output Layer: Softmax activation

Optimizer: Adam (fixed learning rate: 0.001)

Batch Size: 32

Epochs: 20

Accuracy: ~63.8%

🔶 Final Optimized Model
Architecture: 8-layer deep MLP

Hidden layers: 5 dense layers (1024 → 128 units)

Activation: Swish

Residual connection added in 3rd block

Regularization:

Dropout (0.3–0.5)

Batch Normalization

Optimizer: Adam with Exponential Decay learning rate schedule

Batch Size: 64

Epochs: Up to 100 (with EarlyStopping)

Callbacks: EarlyStopping, learning rate scheduler

Accuracy: 83.8%

📊 Feature Engineering
Chess positions encoded using Forsyth–Edwards Notation (FEN)

FEN parsed into a 73-dimensional feature vector:

64 for board layout (piece-to-integer mapping)

1 for turn (white/black)

4 for castling rights

2 for en passant possibility

2 for move counters

🔄 Optimization Strategies
Activation Functions Explored:

Tanh, Leaky ReLU, Swish, SparseMax

Hyperparameter Tuning Techniques:

Manual tuning

Grid Search

Bayesian Optimization

Neural Architecture Search (NAS)

Architectures Tested:

MLP (baseline and optimized)

ResNet-style MLP

MobileNet (via transfer learning)

♟ Stockfish Evaluation
Model predictions were compared against Stockfish evaluations to benchmark accuracy. On a large set:

Average deviation from Stockfish: 178.68

Maximum deviation: 8226 (in complex positions)

For simpler positions: Alignment was strong (avg diff ~29.4)

📈 Evaluation Results

Method	Accuracy (5000 games)

MLP (Initial)	63.8%

MLP (Optimized)	83.8%

ResNet-style MLP	72.92%

MobileNet	84.95% (10 games)

NAS	70.8%

Grid Search	79.56%

Bayesian Optimization	93.33% (10 games)

🧪 Experiments Conducted
Training across varying dataset sizes (10, 1000, 5000 games)

Loss and accuracy tracked across epochs

Comparative experiments across architecture types

Hyperparameter tuning across:

Epochs (20 to 100)

Batch sizes (32, 64)

Optimizers (Adam, SGD)

Learning rates (fixed vs decayed)

Activation functions

🏁 Key Takeaways
MLPs can effectively learn to evaluate chess positions from raw board state encodings.

Swish activation and exponential learning rate decay significantly improved training stability and accuracy.

Performance can rival traditional engines like Stockfish in general position prediction (excluding deep tactical lines).

Use of oneDNN backend and callbacks (e.g., early stopping) helped improve performance and training time on CPUs.
