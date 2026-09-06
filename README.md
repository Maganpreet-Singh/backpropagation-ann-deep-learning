# 🧠 Backpropagation ANN — Deep Learning From Scratch

<p align="center">
  <b>A hands-on implementation of Artificial Neural Networks and Backpropagation using only Python, NumPy, Pandas, and Matplotlib.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-From%20Scratch-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Deep%20Learning-ANN-FF6F00?style=for-the-badge" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab">
</p>

---

## 📌 Overview

This repository is a **from-scratch exploration of how Artificial Neural Networks learn**.

Instead of relying on high-level deep-learning frameworks such as TensorFlow or PyTorch, the notebooks implement the core learning mechanics directly with **NumPy**. The goal is to make the mathematics and flow of a neural network visible rather than hiding everything behind a library call.

The project covers two fundamental supervised-learning problems:

- **Classification** — predict whether a student is placed or not placed.
- **Regression** — predict a student's LPA (salary package) from academic/profile features.

The classification notebook builds a **2 → 2 → 1 neural network** and explicitly implements forward propagation, binary cross-entropy, backpropagation, gradient descent, prediction, and loss tracking. The regression notebook follows the same learning philosophy with a **2 → 2 → 1 feed-forward network** and mean squared error.

> **Core idea:** understand what happens inside a neural network before depending on frameworks that automate it.

---

## 🎯 Learning Objectives

This repository is designed to build strong fundamentals in:

- Artificial Neural Networks (ANNs)
- Forward Propagation
- Backward Propagation / Backpropagation
- Weights and Biases
- Activation Functions
- Sigmoid Function
- Sigmoid Derivative
- Loss Functions
- Binary Cross-Entropy
- Mean Squared Error (MSE)
- Gradients
- Gradient Descent
- Parameter Initialization
- Feature Scaling
- Matrix-based neural-network computation
- Training loops
- Model prediction
- Loss visualization

---

## 🗂️ Repository Structure

```text
backpropagation-ann-deep-learning/
│
├── Backpropagation_for_Classification.ipynb
├── Backpropagation_for_Regression.ipynb
└── README.md
```

### 📓 `Backpropagation_for_Classification.ipynb`

A complete binary-classification example that builds and trains a neural network from scratch.

**Highlights:**
- Small placement dataset using `cgpa` and `profile_score`
- Min-max feature scaling
- Sigmoid activation
- 2 → 2 → 1 architecture
- Binary cross-entropy loss
- Explicit gradient calculation
- Gradient-descent parameter updates
- Prediction thresholding at `0.5`
- Training-loss visualization
- New-student prediction

### 📓 `Backpropagation_for_Regression.ipynb`

A regression example that manually constructs a feed-forward neural network.

**Highlights:**
- Small LPA prediction dataset
- 2 → 2 → 1 architecture
- Linear forward propagation
- Mean squared error loss
- Manual backward propagation
- Gradient-descent updates
- Epoch-wise loss tracking
- Prediction vs. actual comparison
- Error calculation

---

## 🧩 Neural Network Architecture

Both notebooks use a compact network that is intentionally small enough to inspect mathematically.

```text
                Input Layer
              ┌──────────────┐
              │   Feature 1  │
              │   Feature 2  │
              └──────┬───────┘
                     │
                     ▼
              Hidden Layer
              ┌──────────────┐
              │   Neuron 1   │
              │   Neuron 2   │
              └──────┬───────┘
                     │
                     ▼
               Output Layer
              ┌──────────────┐
              │   Prediction │
              └──────────────┘
```

In mathematical form:

### Forward Pass

For the classification network:

```text
Z₁ = W₁X + b₁
A₁ = sigmoid(Z₁)

Z₂ = W₂A₁ + b₂
A₂ = sigmoid(Z₂)
```

where `A₂` is the predicted probability.

### Backward Pass

For the output layer:

```text
dZ₂ = A₂ - y
dW₂ = (dZ₂ · A₁ᵀ) / m
db₂ = sum(dZ₂) / m
```

The error is then propagated toward the hidden layer:

```text
dA₁ = W₂ᵀ · dZ₂
dZ₁ = dA₁ · sigmoid'(A₁)
```

and the first-layer gradients are calculated:

```text
dW₁ = (dZ₁ · Xᵀ) / m
db₁ = sum(dZ₁) / m
```

Finally, gradient descent updates the parameters:

```text
W := W - η · dW
b := b - η · db
```

where `η` is the learning rate.

---

## 🔬 Classification Experiment

The classification notebook uses a tiny student-placement dataset:

| CGPA | Profile Score | Placed |
|---:|---:|---:|
| 8 | 8 | 1 |
| 7 | 9 | 1 |
| 6 | 10 | 0 |
| 5 | 5 | 0 |

The inputs are scaled using min-max normalization before training.

### Training Configuration

```text
Architecture : 2 → 2 → 1
Activation   : Sigmoid
Loss         : Binary Cross-Entropy
Epochs       : 5000
Learning Rate: 1.0
Seed         : 42
```

### Observed Training Result

The recorded notebook run shows the loss falling from approximately **0.7903** at the first epoch to **0.000744** after 5000 epochs.

The same run reaches **100% training accuracy** on the four training examples.

Example prediction:

```text
Input:
CGPA = 7.5
Profile Score = 8.5

Predicted probability of placement ≈ 0.99979
Predicted class = 1
```

> ⚠️ Because this is a deliberately tiny educational dataset, the 100% training accuracy is useful for demonstrating the learning process, not for claiming real-world predictive performance.

---

## 📈 Regression Experiment

The regression notebook uses the following toy dataset:

| CGPA | Profile Score | LPA |
|---:|---:|---:|
| 8 | 8 | 4 |
| 7 | 9 | 5 |
| 6 | 10 | 6 |
| 5 | 12 | 7 |

### Training Configuration

```text
Architecture : 2 → 2 → 1
Activation   : Linear
Loss         : Mean Squared Error (MSE)
Epochs       : 1000
Learning Rate: 0.0001
```

The recorded run reduces the mean epoch loss from approximately **27.8697** at epoch 1 to **0.0900** at epoch 1000.

The final recorded MSE is approximately **0.0878**.

Example outputs from the notebook include:

| Actual LPA | Predicted LPA | Error |
|---:|---:|---:|
| 4.0 | 4.4769 | -0.4769 |
| 5.0 | 5.0811 | -0.0811 |
| 6.0 | 5.6854 | 0.3146 |
| 7.0 | 6.8649 | 0.1351 |

---

## ⚙️ How Backpropagation Works Here

The learning process follows the classic loop:

```text
┌───────────────────────┐
│  Initialize weights   │
│  and biases           │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Forward Propagation   │
│ Calculate prediction  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Compute Loss          │
│ How wrong are we?     │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Backpropagation       │
│ Calculate gradients   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Gradient Descent      │
│ Update W and b        │
└───────────┬───────────┘
            │
            └───────────────► Repeat
```

This is the heart of neural-network training: **make a prediction → measure the error → propagate the error backward → update parameters → repeat**.

---

## 🧮 Key Concepts Implemented From Scratch

### 1. Activation Function

The classification notebook uses the sigmoid function:

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
```

Its derivative is implemented as:

```python
def sigmoid_derivative(a):
    return a * (1.0 - a)
```

### 2. Binary Cross-Entropy

For binary classification:

```text
L = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)]
```

### 3. Mean Squared Error

For regression:

```text
MSE = mean((y - ŷ)²)
```

### 4. Gradient Descent

The parameters are updated in the direction that reduces the loss:

```text
θ := θ - η ∇L(θ)
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **NumPy** | Matrix operations and neural-network mathematics |
| **Pandas** | Dataset creation and tabular analysis |
| **Matplotlib** | Training-loss visualization |
| **Google Colab** | Notebook execution environment |

No TensorFlow, PyTorch, Keras, or scikit-learn model training APIs are used for the neural-network implementation.

---

## 🚀 Getting Started

### Option 1 — Run in Google Colab

The notebooks are designed to work well in Google Colab.

1. Open either notebook on GitHub.
2. Open it in Google Colab.
3. Run the cells from top to bottom.
4. Inspect the intermediate matrices, gradients, loss values, and predictions.

### Option 2 — Run Locally

Clone the repository:

```bash
git clone https://github.com/Maganpreet-Singh/backpropagation-ann-deep-learning.git
cd backpropagation-ann-deep-learning
```

Install the required Python packages:

```bash
pip install numpy pandas matplotlib jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Then open either notebook.

---

## 🧠 Why Build an ANN From Scratch?

Modern frameworks can train a neural network with a few lines of code. That is useful in production, but it can hide the mechanism that makes learning possible.

This project intentionally goes the other way.

By implementing the network manually, you can see:

```text
Input
  ↓
Weighted Sum
  ↓
Activation
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Parameter Update
  ↓
Improved Prediction
```

That understanding becomes valuable later when working with deeper networks, optimizers, CNNs, RNNs, Transformers, and modern deep-learning frameworks.

---

## 📚 What You Can Learn From This Repository

After studying these notebooks, you should have a clearer understanding of:

- How a neuron computes an output
- Why weights and biases are learnable parameters
- How an activation function changes a neural network
- How a loss function measures error
- How the chain rule enables backpropagation
- How gradients tell the network which direction to move
- How gradient descent changes weights and biases
- Why feature scaling can improve gradient-based learning
- How matrix multiplication makes neural-network computation efficient
- How classification and regression differ at the output/loss level

---

## ⚠️ Limitations

This repository is primarily **educational**.

The datasets are intentionally tiny and manually created, so the experiments are not representative of production-grade machine-learning systems.

In particular:

- The classification result is training performance on only four examples.
- The regression experiment uses only four observations.
- There is no train/validation/test split.
- There is no regularization.
- There is no advanced optimizer such as Adam or RMSProp.
- Hyperparameter tuning is intentionally minimal.
- The purpose is to understand the mechanics, not maximize generalization.

That is a feature of the learning exercise, not a bug.

---

## 🔮 Possible Next Steps

Natural extensions of this project include:

- Add ReLU and tanh activations
- Implement softmax for multi-class classification
- Add train/validation/test splits
- Add mini-batch and stochastic gradient descent
- Implement Momentum, RMSProp, and Adam
- Add weight initialization strategies such as Xavier and He initialization
- Add regularization and dropout
- Build deeper multi-layer networks
- Add model serialization and inference pipelines
- Compare the from-scratch implementation with PyTorch/TensorFlow
- Extend the notebooks into reusable Python modules
- Apply the network to a larger real-world dataset

---

## 💡 Project Philosophy

> **Learn the mathematics. Implement the mechanism. Then use the framework.**

The objective is not to reinvent PyTorch.

The objective is to understand enough of the machinery underneath a deep-learning framework that concepts such as gradients, activations, losses, optimizers, and training loops stop feeling like magic.

---

## 👨‍💻 Author

**Maganpreet Singh**

This repository is part of a hands-on journey into **Deep Learning and AI Engineering**, progressing from foundational mathematics and neural networks toward more advanced AI systems.

---

## ⭐ Support

If this repository helped you understand backpropagation or neural networks a little better, consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is intended for learning and educational use.

---

<p align="center">
  <b>From equations → gradients → code → learning.</b>
</p>
