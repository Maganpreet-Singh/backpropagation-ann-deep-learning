<div align="center">

# 🧠 Backpropagation ANN — Deep Learning From Scratch

### Understanding how neural networks actually learn — one equation, one gradient, and one parameter update at a time.

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-From%20Scratch-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Handling-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep%20Learning-ANN-FF6F00?style=for-the-badge" />
</p>

<p>
  <a href="https://github.com/Maganpreet-Singh/backpropagation-ann-deep-learning">Repository</a> •
  <a href="https://github.com/Maganpreet-Singh/backpropagation-ann-deep-learning/blob/main/Backpropagation_for_Classification.ipynb">Classification Notebook</a> •
  <a href="https://github.com/Maganpreet-Singh/backpropagation-ann-deep-learning/blob/main/Backpropagation_for_Regression.ipynb">Regression Notebook</a>
</p>

</div>

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Why This Project?](#-why-this-project)
- [What Is Implemented](#-what-is-implemented)
- [Repository Structure](#-repository-structure)
- [Neural Network Architecture](#-neural-network-architecture)
- [How a Neural Network Learns](#-how-a-neural-network-learns)
- [Forward Propagation](#-forward-propagation)
- [Loss Functions](#-loss-functions)
- [Backpropagation](#-backpropagation)
- [Gradient Descent](#-gradient-descent)
- [Classification Experiment](#-classification-experiment)
- [Regression Experiment](#-regression-experiment)
- [Training Results](#-training-results)
- [Important Concepts](#-important-concepts)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [How to Study This Repository](#-how-to-study-this-repository)
- [Design Choices](#-design-choices)
- [Limitations](#-limitations)
- [Possible Improvements](#-possible-improvements)
- [Learning Roadmap](#-learning-roadmap)
- [Key Takeaways](#-key-takeaways)
- [Author](#-author)

---

## 🚀 About the Project

**Backpropagation ANN — Deep Learning From Scratch** is a hands-on implementation of the core mechanics behind Artificial Neural Networks (ANNs), written without using high-level neural-network training APIs.

The project is built to answer a simple but important question:

> **What is actually happening inside a neural network when it learns?**

Instead of calling a framework such as `model.fit()`, the notebooks explicitly implement the important pieces of the learning process using **NumPy**:

```text
Input Data
    ↓
Weighted Sum
    ↓
Activation
    ↓
Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Gradient Calculation
    ↓
Gradient Descent
    ↓
Updated Parameters
    ↓
Repeat
```

The repository demonstrates the process through two supervised-learning tasks:

1. **Binary Classification** — predicting whether a student is placed.
2. **Regression** — predicting a student's LPA / salary package.

The datasets are deliberately tiny and easy to inspect. That is intentional: this repository is a **learning laboratory**, not a production machine-learning system.

---

## 🎯 Why This Project?

Modern deep-learning frameworks are incredibly powerful, but they can hide the mathematical machinery underneath the API.

A single line such as:

```python
model.fit(X, y)
```

can represent a huge amount of work:

- parameter initialization
- matrix multiplication
- activation functions
- loss calculation
- chain rule
- partial derivatives
- gradient propagation
- parameter updates
- repeated optimization

This project pulls those layers apart.

The goal is not to replace PyTorch or TensorFlow. The goal is to understand **why those frameworks work**.

Once the fundamentals are clear, advanced topics such as CNNs, RNNs, Transformers, attention mechanisms, optimizers, and modern AI systems become much easier to reason about.

---

## 🧩 What Is Implemented?

### Core Neural Network Concepts

- Artificial Neural Networks
- Neurons
- Layers
- Weights
- Biases
- Forward propagation
- Backward propagation
- Backpropagation
- Gradient calculation
- Chain-rule-based differentiation
- Gradient descent
- Parameter updates
- Training loops
- Prediction

### Activation & Loss Concepts

- Sigmoid activation
- Sigmoid derivative
- Linear output
- Binary Cross-Entropy
- Mean Squared Error
- Prediction thresholding

### Numerical & Data Concepts

- NumPy arrays
- Matrix multiplication
- Vectorized calculations
- Matrix shape management
- Min-max feature scaling
- Training-loss tracking
- Loss visualization

---

## 📂 Repository Structure

```text
backpropagation-ann-deep-learning/
│
├── 📓 Backpropagation_for_Classification.ipynb
│   └── Binary classification ANN implemented from scratch
│
├── 📓 Backpropagation_for_Regression.ipynb
│   └── Regression ANN implemented from scratch
│
└── 📄 README.md
    └── Project documentation
```

---

# 🧠 Neural Network Architecture

Both notebooks intentionally use a compact **2 → 2 → 1** network so that every calculation can be inspected.

```text
                         INPUT LAYER
                    ┌───────────────────┐
                    │                   │
                    │   Feature 1       │
                    │   Feature 2       │
                    │                   │
                    └─────────┬─────────┘
                              │
                              │ W₁, b₁
                              ▼
                        HIDDEN LAYER
                    ┌───────────────────┐
                    │                   │
                    │    Neuron 1       │
                    │    Neuron 2       │
                    │                   │
                    └─────────┬─────────┘
                              │
                              │ W₂, b₂
                              ▼
                         OUTPUT LAYER
                    ┌───────────────────┐
                    │                   │
                    │    Prediction     │
                    │                   │
                    └───────────────────┘
```

For classification:

```text
2 Inputs → 2 Hidden Neurons → 1 Output Probability
```

For regression:

```text
2 Inputs → 2 Hidden Units → 1 Continuous Output
```

The small architecture makes the project ideal for manually tracing the data flow and derivatives.

---

# 🔄 How a Neural Network Learns

Training can be understood as four major stages.

### 1️⃣ Forward Propagation

Use the current weights and biases to calculate a prediction.

### 2️⃣ Loss Calculation

Compare the prediction against the real target.

### 3️⃣ Backpropagation

Determine how much each parameter contributed to the error.

### 4️⃣ Gradient Descent

Move the parameters in the direction that reduces the loss.

Then repeat the process many times.

```text
              ┌─────────────────────┐
              │ Initialize Params   │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │ Forward Propagation │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   Compute Loss      │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │  Backpropagation    │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │ Gradient Descent    │
              └──────────┬──────────┘
                         │
                         └──────────► Repeat
```

---

# ➡️ Forward Propagation

Forward propagation is the process of sending input data through the network to produce an output.

For the classification notebook, the first layer computes:

```text
Z₁ = W₁X + b₁
```

The hidden activation is:

```text
A₁ = sigmoid(Z₁)
```

Then the second layer computes:

```text
Z₂ = W₂A₁ + b₂
```

and the final activation is:

```text
A₂ = sigmoid(Z₂)
```

The final value `A₂` is interpreted as the probability of the positive class.

### Sigmoid Function

The sigmoid function maps any real number into the range `(0, 1)`:

```text
              1
σ(z) = ─────────────
          1 + e⁻ᶻ
```

In the notebook:

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
```

The implementation uses clipping inside the exponential to avoid numerical overflow for very large positive or negative values.

---

# 📉 Loss Functions

A network needs a way to measure how wrong its predictions are.

That measurement is called the **loss**.

Different tasks use different loss functions.

---

## 🔵 Binary Cross-Entropy — Classification

For binary classification, the notebook uses binary cross-entropy:

```text
L = -(1/m) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]
```

where:

- `y` = actual label
- `ŷ` = predicted probability
- `m` = number of examples

The classification implementation clips the prediction probabilities slightly before taking logarithms for numerical stability.

The lower the loss, the closer the predicted probabilities are to the correct labels.

---

## 🟢 Mean Squared Error — Regression

For regression, the notebook uses mean squared error:

```text
MSE = mean((y - ŷ)²)
```

This penalizes larger prediction errors more strongly because the difference is squared.

---

# 🔙 Backpropagation

Backpropagation is the mechanism that allows the network to learn from its mistakes.

The basic idea is:

> **Start at the loss, move backward through the network, and calculate how the loss changes with respect to every parameter.**

That is essentially the chain rule applied repeatedly through the computational graph.

---

## Output Layer Gradient

For the classification network, the output gradient is implemented as:

```text
dZ₂ = A₂ - y
```

Then:

```text
dW₂ = (dZ₂ · A₁ᵀ) / m
```

and:

```text
db₂ = sum(dZ₂) / m
```

---

## Hidden Layer Gradient

The gradient is propagated back into the hidden layer:

```text
dA₁ = W₂ᵀ · dZ₂
```

Then the sigmoid derivative is applied:

```text
dZ₁ = dA₁ · sigmoid'(A₁)
```

Finally:

```text
dW₁ = (dZ₁ · Xᵀ) / m
```

and:

```text
db₁ = sum(dZ₁) / m
```

The notebook explicitly stores these gradients and prints their shapes and values so the backward pass can be inspected rather than treated as a black box.

---

# ⬇️ Gradient Descent

Once the gradients are known, the parameters are updated.

The general rule is:

```text
θ ← θ - η ∇L(θ)
```

where:

- `θ` = parameter
- `η` = learning rate
- `∇L(θ)` = gradient of the loss

In practical terms:

```python
parameters['W1'] -= learning_rate * grads['dW1']
parameters['b1'] -= learning_rate * grads['db1']
parameters['W2'] -= learning_rate * grads['dW2']
parameters['b2'] -= learning_rate * grads['db2']
```

The entire learning system therefore becomes a repeating feedback loop:

```text
Prediction
   ↓
Loss
   ↓
Gradient
   ↓
Parameter Update
   ↓
Better Prediction
```

---

# 🧪 Classification Experiment

## Problem Statement

The classification notebook demonstrates how an ANN can learn a binary target:

> **Is a student placed?**

The input features are:

- `cgpa`
- `profile_score`

The target is:

- `placed`

where:

```text
1 → Placed
0 → Not Placed
```

---

## Dataset

The notebook creates the following toy dataset:

| CGPA | Profile Score | Placed |
|---:|---:|---:|
| 8 | 8 | 1 |
| 7 | 9 | 1 |
| 6 | 10 | 0 |
| 5 | 5 | 0 |

This dataset is intentionally tiny so that every calculation remains easy to inspect.

---

## Feature Preparation

The classification notebook converts the feature matrix into the shape required for matrix-based neural-network calculations:

```text
X.shape = (number_of_features, number_of_examples)
```

For this dataset:

```text
X.shape = (2, 4)
y.shape = (1, 4)
```

The notebook also demonstrates min-max scaling:

```text
X_scaled = (X - X_min) / (X_max - X_min + ε)
```

Feature scaling is useful for gradient-based optimization because it keeps the input values within a more manageable numerical range.

---

## Model Configuration

```text
Architecture : 2 → 2 → 1
Activation   : Sigmoid
Loss         : Binary Cross-Entropy
Epochs       : 5000
Learning Rate: 1.0
Seed         : 42
Threshold    : 0.5
```

---

## Training Process

The notebook performs the following operations:

```text
1. Initialize W₁, b₁, W₂, b₂
2. Scale input features
3. Run forward propagation
4. Calculate binary cross-entropy
5. Run backpropagation
6. Calculate dW and db
7. Update parameters with gradient descent
8. Store the loss
9. Repeat for 5000 epochs
```

---

## Classification Results

The recorded notebook run starts with:

```text
Epoch     1 | Loss = 0.790275
```

and reaches:

```text
Epoch  5000 | Loss = 0.000744
```

The recorded training accuracy is:

```text
Training accuracy: 100.00%
```

The trained network produces probabilities such as:

| CGPA | Profile Score | Actual | Predicted Probability | Predicted Class |
|---:|---:|---:|---:|---:|
| 8 | 8 | 1 | 0.9999 | 1 |
| 7 | 9 | 1 | 0.9985 | 1 |
| 6 | 10 | 0 | 0.0011 | 0 |
| 5 | 5 | 0 | 0.0003 | 0 |

---

## Example Prediction

The notebook also tests the model on a new student:

```text
CGPA           = 7.5
Profile Score  = 8.5
```

Recorded output:

```text
Probability ≈ 0.9997938
Predicted class = 1
Placed
```

Again, this should be interpreted as a demonstration of the training mechanism, not as a meaningful real-world placement model.

---

# 📊 Regression Experiment

## Problem Statement

The regression notebook demonstrates a continuous prediction problem:

> **Predict LPA from CGPA and profile score.**

The input features are:

- `cgpa`
- `profile_score`

The target is:

- `lpa`

---

## Dataset

The notebook uses:

| CGPA | Profile Score | LPA |
|---:|---:|---:|
| 8 | 8 | 4 |
| 7 | 9 | 5 |
| 6 | 10 | 6 |
| 5 | 12 | 7 |

Again, the dataset is deliberately small for educational transparency.

---

## Model Configuration

```text
Architecture : 2 → 2 → 1
Output       : Linear
Loss         : Mean Squared Error
Epochs       : 1000
Learning Rate: 0.0001
```

Unlike the classification notebook, the regression notebook uses a linear output, allowing the network to predict continuous values rather than probabilities.

---

## Regression Training

The notebook demonstrates a complete training step on individual examples and then extends that idea to a training loop.

At each epoch:

```text
For each training example:
    ↓
Forward pass
    ↓
Calculate MSE
    ↓
Backward pass
    ↓
Calculate gradients
    ↓
Update parameters
```

The loss is averaged across the examples to monitor training progress.

---

## Regression Results

The recorded run shows the mean loss decreasing from:

```text
Epoch    1 | Loss: 27.869678
```

to:

```text
Epoch 1000 | Loss: 0.089969
```

The final recorded MSE is approximately:

```text
0.0877978
```

Example predictions from the trained network:

| CGPA | Profile Score | Actual LPA | Predicted LPA | Error |
|---:|---:|---:|---:|---:|
| 8 | 8 | 4.0 | 4.4769 | -0.4769 |
| 7 | 9 | 5.0 | 5.0811 | -0.0811 |
| 6 | 10 | 6.0 | 5.6854 | 0.3146 |
| 5 | 12 | 7.0 | 6.8649 | 0.1351 |

The error column is calculated as:

```text
error = actual - predicted
```

---

# 🔬 Classification vs Regression

The two notebooks highlight an important distinction in supervised learning.

| Aspect | Classification | Regression |
|---|---|---|
| Goal | Predict a class | Predict a continuous value |
| Example | Placed / Not Placed | LPA |
| Output | Probability | Numeric value |
| Output activation | Sigmoid | Linear |
| Loss | Binary Cross-Entropy | Mean Squared Error |
| Prediction rule | Threshold at 0.5 | Direct numeric output |

The underlying learning idea remains the same:

```text
Forward → Loss → Backpropagation → Update → Repeat
```

What changes is the output interpretation and the loss function.

---

# 🧱 Parameter Shapes

A major part of understanding neural networks is understanding matrix dimensions.

For the classification network:

```text
W₁ : (2, 2)
b₁ : (2, 1)
W₂ : (1, 2)
b₂ : (1, 1)
```

For input data:

```text
X : (2, 4)
y : (1, 4)
```

This makes matrix multiplication possible:

```text
W₁ @ X
```

which gives:

```text
(2, 2) × (2, 4) = (2, 4)
```

Then:

```text
W₂ @ A₁
```

becomes:

```text
(1, 2) × (2, 4) = (1, 4)
```

Thinking in shapes is one of the most important practical skills when implementing neural networks with NumPy.

---

# 🧮 Key Mathematical Building Blocks

## Weighted Sum

Every neuron begins with a weighted combination of its inputs:

```text
z = w₁x₁ + w₂x₂ + ... + b
```

or, in vector form:

```text
z = Wx + b
```

---

## Activation

The weighted sum is passed through an activation function.

For sigmoid:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

---

## Prediction

The final layer transforms the information accumulated by earlier layers into the model's prediction.

---

## Loss

The loss quantifies the difference between prediction and target.

---

## Gradient

The gradient tells us how changing a parameter will change the loss.

---

## Update

Gradient descent uses that information to move parameters toward lower loss.

---

# 🔍 Functions Implemented in the Notebooks

The classification notebook contains explicit implementations for functions such as:

```python
sigmoid()
sigmoid_derivative()
initialize_parameters()
forward_propagation()
compute_loss()
backward_propagation()
update_parameters()
train()
predict()
```

The regression notebook similarly builds its own components, including:

```python
initialize_parameters()
linear_forward()
L_layer_forward()
mse_loss()
backward_propagation()
update_parameters()
train()
```

The important point is that these are not merely called from a deep-learning framework. The learning logic itself is written explicitly.

---

# 🧰 Technologies Used

| Technology | Role in Project |
|---|---|
| 🐍 **Python** | Programming language |
| 🔢 **NumPy** | Matrix operations, gradients, numerical computation |
| 🐼 **Pandas** | Dataset construction and result tables |
| 📊 **Matplotlib** | Loss-curve visualization |
| ☁️ **Google Colab** | Notebook execution environment |
| 📓 **Jupyter Notebooks** | Interactive experimentation |

### Intentionally Not Used for Model Training

- TensorFlow
- PyTorch
- Keras
- Scikit-learn model training APIs

The point of this repository is to implement the mechanics manually.

---

# 🚀 Getting Started

## Option 1 — Google Colab

The easiest way to explore the project is through Google Colab.

1. Open either notebook from the repository.
2. Open the notebook in Colab.
3. Run the cells from top to bottom.
4. Read the comments and inspect each intermediate calculation.
5. Modify the weights, learning rate, epochs, or input data and observe what changes.

---

## Option 2 — Run Locally

### Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/backpropagation-ann-deep-learning.git
cd backpropagation-ann-deep-learning
```

### Install dependencies

```bash
pip install numpy pandas matplotlib jupyter
```

### Start Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
Backpropagation_for_Classification.ipynb
```

or:

```text
Backpropagation_for_Regression.ipynb
```

---

# 🧭 How to Study This Repository

Do not just run the notebooks and look at the final number.

The best way to use this project is to trace the learning process manually.

### Step 1 — Understand the data

Look at the shape and meaning of `X` and `y`.

### Step 2 — Inspect the parameters

Look at every `W` and `b` matrix.

Ask:

> What does each number represent?

### Step 3 — Follow the forward pass

Track:

```text
X → Z₁ → A₁ → Z₂ → A₂
```

### Step 4 — Inspect the loss

Ask:

> Why is this loss high or low?

### Step 5 — Follow the backward pass

Track:

```text
dZ₂ → dW₂ / db₂ → dA₁ → dZ₁ → dW₁ / db₁
```

### Step 6 — Watch the update

Compare the parameters before and after gradient descent.

### Step 7 — Watch the loss curve

A successful optimization process should generally move toward lower loss for this simple training setup.

---

# 🎨 Visualizing Training

Both notebooks generate training-loss plots using Matplotlib.

The purpose of the plot is to see whether the optimization process is actually improving the model.

Conceptually:

```text
Loss
 │\
 │ \
 │  \
 │   \      
 │    \___
 │        \____
 └────────────────── Epochs
```

A downward trend means the model is finding parameter values that reduce the training objective.

The exact curve depends on initialization, learning rate, architecture, data, and the optimization process.

---

# 🧠 Important Concepts to Understand Before Moving On

This project is a strong foundation for the following deeper topics:

### Neural Network Fundamentals

- neurons
- layers
- weights
- biases
- activations
- output layers

### Optimization

- gradients
- learning rate
- gradient descent
- convergence
- optimization stability

### Deep Learning Mathematics

- derivatives
- partial derivatives
- chain rule
- matrix calculus
- vectorization

### Model Training

- epochs
- loss curves
- prediction
- parameter updates
- inference

---

# ⚙️ Design Choices

## Why NumPy?

NumPy makes the underlying mathematics visible while still providing efficient array operations.

The project therefore sits in a useful middle ground:

```text
Pure Python loops
       ↓
   NumPy arrays
       ↓
Deep Learning Frameworks
```

You get to work with the same matrix-oriented ideas used by larger frameworks without immediately hiding them behind abstraction.

---

## Why a Tiny Dataset?

A large dataset would make the mathematics harder to inspect.

With four examples, it is possible to:

- print the inputs
- print the targets
- inspect the activations
- inspect the gradients
- inspect parameter updates
- verify predictions manually

This is ideal for learning the mechanics.

---

## Why Two Separate Notebooks?

Classification and regression share the same broad learning pipeline but differ in their output behavior and loss functions.

Separating them makes those differences easier to see.

---

# ⚠️ Limitations

This is an **educational implementation**, not a production-ready machine-learning package.

### Dataset Limitations

The notebooks use only four observations per task. That is nowhere near enough to establish meaningful real-world predictive performance.

### Evaluation Limitations

The project does not currently include:

- train/validation/test splits
- cross-validation
- confusion matrix
- precision/recall/F1
- ROC-AUC
- robust regression metrics
- statistical significance testing

### Modeling Limitations

The implementation does not currently provide:

- dropout
- batch normalization
- regularization
- early stopping
- Adam
- RMSProp
- momentum
- learning-rate scheduling
- advanced initialization schemes
- automated hyperparameter tuning

### Engineering Limitations

The project is notebook-based rather than packaged as a reusable library or deployment service.

These limitations are appropriate for the project's current purpose: **learning the fundamentals of neural-network training**.

---

# 🔮 Possible Improvements

A natural next version could evolve this project significantly.

## Neural Network Improvements

- Add ReLU
- Add tanh
- Add softmax
- Add configurable hidden layers
- Support arbitrary network depth
- Add configurable output dimensions

## Optimization Improvements

- Stochastic Gradient Descent
- Mini-batch Gradient Descent
- Momentum
- RMSProp
- Adam
- Learning-rate decay

## Initialization Improvements

- Xavier initialization
- He initialization
- Better random initialization strategies

## Generalization Improvements

- Train/validation/test splitting
- L2 regularization
- Dropout
- Early stopping

## Evaluation Improvements

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- MAE
- MSE
- RMSE
- R²

## Engineering Improvements

- Move reusable functions into `.py` modules
- Add unit tests
- Add type hints
- Add documentation strings
- Add experiment configuration
- Save learned parameters
- Build a reusable inference pipeline
- Add command-line execution
- Compare with PyTorch implementations

---

# 🗺️ Learning Roadmap

This repository naturally fits into a larger Deep Learning progression:

```text
Python
  ↓
NumPy
  ↓
Linear Algebra
  ↓
Calculus
  ↓
Perceptron
  ↓
Artificial Neural Networks
  ↓
Forward Propagation
  ↓
Backpropagation
  ↓
Gradient Descent
  ↓
Deep Neural Networks
  ↓
Optimizers
  ↓
Regularization
  ↓
CNNs
  ↓
RNNs / LSTMs
  ↓
Attention
  ↓
Transformers
  ↓
LLMs
  ↓
RAG
  ↓
AI Agents
  ↓
Deployment & MLOps
```

The specific purpose of this repository is to make the transition from **basic neural-network theory to actual implementation**.

---

# 💡 Key Takeaways

After working through the notebooks, you should be able to explain the following in your own words:

### What is a weight?

A learnable parameter that controls the influence of an input on a neuron.

### What is a bias?

A learnable offset that gives the neuron additional flexibility.

### What is forward propagation?

The process of using the current parameters to transform input data into predictions.

### What is a loss function?

A mathematical measurement of how far the prediction is from the target.

### What is backpropagation?

A systematic way of calculating how the loss changes with respect to network parameters by propagating gradients backward through the computation graph.

### What is gradient descent?

An optimization method that updates parameters in the direction that reduces the loss.

### Why does training work?

Because repeated gradient-based updates can move the model toward parameter values that fit the training data better.

---

# 🧪 Experiments You Can Try

Once you understand the existing notebooks, modify them.

### Experiment 1 — Change the learning rate

Try different values and observe whether the loss decreases faster, slower, or becomes unstable.

### Experiment 2 — Change the number of hidden units

Try:

```text
2 → 1 → 1
2 → 2 → 1
2 → 4 → 1
2 → 8 → 1
```

Then compare the behavior.

### Experiment 3 — Change the initialization seed

Different initial parameters can lead to different optimization trajectories.

### Experiment 4 — Add another feature

Change the architecture and matrix shapes accordingly.

### Experiment 5 — Change the classification threshold

Compare:

```text
threshold = 0.3
threshold = 0.5
threshold = 0.7
```

### Experiment 6 — Replace sigmoid

Try implementing another activation function and examine how the training process changes.

---

# 🏗️ Project Maturity Path

This repository is intentionally at the **fundamentals stage**.

A strong progression from here would be:

```text
Current
│
├── ANN from scratch
├── Backpropagation
├── Gradient Descent
│
▼
Next
│
├── Deeper ANN
├── ReLU / Softmax
├── Mini-batches
├── Adam
├── Regularization
│
▼
Advanced
│
├── CNN
├── RNN / LSTM
├── Attention
├── Transformers
│
▼
Applied AI
│
├── RAG
├── LLM Applications
├── AI Agents
├── APIs
├── Deployment
└── MLOps
```

---

# 📜 Learning Philosophy

> **Do not rush to the framework before understanding the mechanism.**

A neural network should not feel like magic.

It is a sequence of mathematical operations:

```text
Matrices
   +
Weights
   +
Biases
   ↓
Activation
   ↓
Prediction
   ↓
Loss
   ↓
Derivatives
   ↓
Gradients
   ↓
Optimization
```

Once you can follow that chain, high-level deep-learning libraries become tools rather than mysteries.

---

# 👨‍💻 Author

## Maganpreet Singh

This project is part of a broader hands-on journey into **Deep Learning and AI Engineering**.

The focus is on building concepts from the ground up, turning mathematical ideas into working implementations, and gradually progressing toward modern AI systems.

---

# ⭐ Contributing & Feedback

This repository is primarily a personal learning project, but suggestions, improvements, corrections, and educational discussions are welcome.

Good ways to improve the project include:

- finding mathematical mistakes
- improving numerical stability
- suggesting better explanations
- adding tests
- extending the network architecture
- adding new experiments

---

# 📌 Final Note

This repository is small by design.

The dataset is small.

The network is small.

The number of neurons is small.

But the ideas inside it are fundamental to a huge part of modern machine learning.

Understanding these fundamentals now makes the more complicated systems later much less intimidating.

```text
Learn the equation.
Understand the gradient.
Implement the algorithm.
Then use the framework.
```

<div align="center">

### 🧠 From equations → gradients → code → learning.

**⭐ Star the repository if it helped you understand backpropagation.**

</div>
