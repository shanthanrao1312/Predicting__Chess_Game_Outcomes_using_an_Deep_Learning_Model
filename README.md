# 🧠 Predicting Chess Game Outcomes Using Deep Learning (MLP)

## 📌 Overview

This project explores the use of **deep learning models**, specifically **Multilayer Perceptrons (MLPs)**, to predict the outcome of a chess position—win, draw, or loss—based purely on the board state. The pipeline involves parsing a dataset of millions of chess games into **FEN strings**, converting them into numerical feature vectors, and training a neural network on this representation.

---

![Chess Prediction Visualization](https://github.com/user-attachments/assets/140d9918-f429-46a7-824b-afe167d087eb)

---

## 🧰 Technologies & Frameworks Used

- **TensorFlow (with Keras API)**: Core deep learning framework
- **Intel oneDNN**: CPU-accelerated deep learning operations
- **NumPy, Pandas**: Data processing and transformation
- **Matplotlib**: Visualization of training performance
- **Stockfish**: Chess engine used for comparative evaluation

---

## 🧠 Neural Network Architecture

### 🔷 Initial Model

- **Input Layer**: 73-dimensional feature vector (encoded FEN)
- **Hidden Layers**: 2 layers with 64 neurons each
- **Activation**: ReLU
- **Output Layer**: Softmax activation
- **Optimizer**: Adam (fixed learning rate: 0.001)
- **Batch Size**: 32
- **Epochs**: 20
- **Accuracy**: ~63.8%

### 🔶 Final Optimized Model

- **Architecture**: 8-layer deep MLP
- **Hidden Layers**: 5 dense layers (1024 → 128 units)
- **Activation**: Swish
- **Residual Connection**: Added in the 3rd block
- **Regularization**:
  - Dropout (0.3–0.5)
  - Batch Normalization
- **Optimizer**: Adam with Exponential Decay learning rate schedule
- **Batch Size**: 64
- **Epochs**: Up to 100 (with EarlyStopping)
- **Callbacks**: EarlyStopping, learning rate scheduler
- **Accuracy**: 83.8%

---

## 📊 Feature Engineering

The chess positions were encoded using **Forsyth–Edwards Notation (FEN)** and parsed into a **73-dimensional feature vector**:

1. **64**: Board layout (piece-to-integer mapping)
2. **1**: Turn (white/black)
3. **4**: Castling rights
4. **2**: En passant possibility
5. **2**: Move counters

---

## 🔄 Optimization Strategies

### Activation Functions Explored

- Tanh
- Leaky ReLU
- Swish
- SparseMax

### Hyperparameter Tuning Techniques

- Manual tuning
- Grid Search
- Bayesian Optimization
- Neural Architecture Search (NAS)

### Architectures Tested

- **MLP**: Baseline and optimized versions
- **ResNet-style MLP**
- **MobileNet**: Via transfer learning

---

## ♟ Stockfish Evaluation

Model predictions were compared against **Stockfish** evaluations to benchmark accuracy. Results on a large dataset:

- **Average Deviation from Stockfish**: 178.68
- **Maximum Deviation**: 8226 (in complex positions)
- **Simpler Positions**: Strong alignment (average deviation ~29.4)

---

## 📈 Evaluation Results

| **Method**            | **Accuracy (5000 games)** |
|------------------------|---------------------------|
| MLP (Initial)          | 63.8%                    |
| MLP (Optimized)        | 83.8%                    |
| ResNet-style MLP       | 72.92%                   |
| MobileNet              | 84.95% (10 games)        |
| NAS                    | 70.8%                    |
| Grid Search            | 79.56%                   |
| Bayesian Optimization  | 93.33% (10 games)        |

---

## 🧪 Experiments Conducted

- **Dataset Sizes**: Training across varying sizes (10, 1000, 5000 games)
- **Metrics Tracked**: Loss and accuracy across epochs
- **Comparative Architectures**: MLP, ResNet-style MLP, MobileNet
- **Hyperparameter Tuning**:
  - **Epochs**: 20 to 100
  - **Batch Sizes**: 32, 64
  - **Optimizers**: Adam, SGD
  - **Learning Rates**: Fixed vs decayed
  - **Activation Functions**: Explored various options

---

## 🏁 Key Takeaways

1. **MLPs** can effectively learn to evaluate chess positions from raw board state encodings.
2. **Swish activation** and **exponential learning rate decay** significantly improved training stability and accuracy.
3. Performance rivals traditional engines like **Stockfish** in general position prediction (excluding deep tactical lines).
4. Use of **oneDNN backend** and **callbacks** (e.g., early stopping) improved performance and shortened training time on CPUs.

---

Thank you for visiting this repository! Feel free to explore, contribute, or share your feedback. 😊
