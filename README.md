# ♟️ **Predicting Chess Game Outcomes Using Deep Learning (MLP)** 🧠

Welcome to the fascinating world of **AI Chess Prediction**! This project leverages the power of **Multilayer Perceptrons (MLPs)** to predict chess outcomes—**win**, **draw**, or **loss**—using raw board states. By processing millions of games into **FEN strings** and encoding them into feature vectors, we’ve built highly optimized neural networks that decode chess positions with remarkable precision.

---

<div align="center">
  <img src="https://github.com/user-attachments/assets/140d9918-f429-46a7-824b-afe167d087eb" alt="Chess Prediction Visualization" width="300">
</div>

---

## 🛠️ **Technologies Used**
- 🔸 **TensorFlow (Keras API)**: Core deep learning framework.  
- 🔸 **Intel oneDNN**: CPU-accelerated deep learning operations.  
- 🔸 **NumPy & Pandas**: Efficient data processing pipelines.  
- 🔸 **Matplotlib**: Visualizing training performance.  
- 🔸 **Stockfish**: Comparing model predictions with the world-renowned chess engine.  

---

## 🧠 **Model Architecture**

### 🔷 **Baseline Model**
- **Input**: 73-dimensional FEN-based encoded vector.
- **Hidden Layers**: 2 dense layers with 64 neurons each.
- **Activation**: ReLU.
- **Output**: Softmax activation for 3 classes (win, draw, loss).
- **Optimizer**: Adam (fixed learning rate: 0.001).
- **Performance**: Achieved **63.8% accuracy**.

---

### 🔶 **Optimized Model**
- **Architecture**: 8-layer deep MLP with advanced optimizations.
- **Hidden Layers**: 5 dense layers (1024 → 128 units).
- **Activation**: Swish (for faster convergence).
- **Enhancements**:
  - Residual connections added in the 3rd block.
  - Dropout (0.3–0.5) and Batch Normalization for regularization.
- **Optimizer**: Adam with exponential learning rate decay.
- **Performance**: Achieved **83.8% accuracy**.

---

## 📊 **Feature Engineering**
Chess board states were encoded using **Forsyth–Edwards Notation (FEN)**, which were then parsed into **73-dimensional feature vectors**:
1. **64**: Encoded board layout (piece-to-integer mapping).  
2. **1**: Turn indicator (white or black).  
3. **4**: Castling rights.  
4. **2**: En passant possibilities.  
5. **2**: Move counters (half-move clock & full-move number).  

---

## 🔄 **Optimization Strategies**

### **Activation Functions Explored**
- ReLU  
- Leaky ReLU  
- Swish  
- SparseMax  

### **Hyperparameter Tuning**
- **Methods**: Manual tuning, Grid Search, Bayesian Optimization, Neural Architecture Search (NAS).  
- **Explored Parameters**:
  - **Batch Sizes**: 32, 64.  
  - **Learning Rates**: Fixed vs Decayed.  
  - **Optimizers**: Adam, SGD.  
  - **Epochs**: 20–100 with Early Stopping.  

---

## ♟ **Stockfish Benchmarking**
Model predictions were compared against **Stockfish** evaluations to assess accuracy and deviation:
- **Average Deviation**: 178.68 (lower is better).  
- **Simpler Positions**: Strong alignment (~29.4 deviation).  
- **Complex Positions**: Higher deviation (~8226).  

---

## 📈 **Evaluation Results**

| **Method/Model**       | **Accuracy (5000 Games)** |
|-------------------------|---------------------------|
| **MLP (Baseline)**      | 63.8%                    |
| **MLP (Optimized)**      | 83.8%                    |
| **ResNet-style MLP**    | 72.92%                   |
| **MobileNet**           | 84.95% (10 games)        |
| **Bayesian Optimization**| 93.33% (10 games)        |
| **NAS**                 | 70.8%                    |
| **Grid Search**         | 79.56%                   |

---

## 🧪 **Experiments Conducted**
- **Dataset Sizes**: Evaluated on 10, 1000, 5000 games.  
- **Metrics Tracked**: Loss, accuracy, and deviation from Stockfish.  
- **Comparative Architectures**: MLP, ResNet-style MLP, MobileNet.  
- **Training Tools Used**:
  - Learning rate scheduling.  
  - Early stopping to prevent overfitting.  

---

## 🏆 **Key Takeaways**
1. **MLPs** can effectively predict chess outcomes based on raw FEN encodings.  
2. **Swish activation** and **exponential learning rate decay** significantly improved training stability and accuracy.  
3. Model performance aligns closely with **Stockfish** in simpler positions.  
4. Leveraging **Intel oneDNN** accelerated training on CPUs, making this approach scalable.  

---

💡 **Ready to explore the future of AI in chess?**  
Feel free to contribute, share feedback, or dive into the code. Let's push the boundaries of innovation together! ♟✨
