# Implementation-of-a-Deep-Neural-Network-for-ECG-processing
"A Deep Neural Network framework built from scratch using native MATLAB matrix operations (no toolboxes). Features manual backpropagation derivation, custom optimizers, and pruning for ECG signal classification. 🏆 Awarded Highest Honors."

# Deep Neural Network Implementation from Scratch (MATLAB)

![MATLAB](https://img.shields.io/badge/Core-MATLAB_Native-orange?style=for-the-badge&logo=mathworks)
![Status](https://img.shields.io/badge/Status-Research_Thesis-blue?style=for-the-badge)
![Award](https://img.shields.io/badge/Award-Highest_Honors_(Top_5%25)-gold?style=for-the-badge)

> **⚠️ Language Note:** All source code, comments, and internal documentation are written in **Spanish** to comply with the academic submission requirements of the University of Málaga (Spain). This README provides an English overview for international researchers.

## 📌 Project Overview

This repository hosts the **Deep Neural Network (DNN)** engine developed for my Bachelor's Thesis in Telecommunication Engineering. Unlike standard implementations that rely on "black-box" frameworks (TensorFlow, PyTorch, Keras), this project implements a deep learning architecture **entirely from first principles** using native MATLAB matrix operations.

**The Goal:** To mathematically derive and implement the backpropagation calculus, optimization algorithms, and regularization techniques to classify **1D ECG (Electrocardiogram)** biosignals for arrhythmia detection.

**Key Achievement:** The project was awarded **"Matrícula de Honor" (Highest Honors)** for its technical depth, optimizing the trade-off between mathematical rigor and computational efficiency on the **MIT-BIH PhysioNet** database.

---

## 🧠 Mathematical Foundations

The core of this engine is the manual derivation of the gradient descent algorithms. No auto-differentiation libraries were used.

### 1. Forward Propagation (Vectorized)
For any given layer $l$, the linear aggregation and activation are computed as:
$$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$
*Implemented Activations:* ReLU (Hidden Layers) and Sigmoid (Output Layer).

### 2. Loss Function
The model minimizes the **Binary Cross-Entropy** cost function, derived from the Maximum Likelihood Estimation principle:
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)}) \right]$$

### 3. Backpropagation (The Engine)
Gradients are computed via the **Multivariate Chain Rule**. The error term $\delta^{[l]}$ is propagated backwards from the output layer to the input:
$$\delta^{[L]} = A^{[L]} - Y$$
$$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot g'^{[l]}(Z^{[l]})$$
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} \delta^{[l]} (A^{[l-1]})^T$$

---

## 🛠️ Features & Algorithms

This isn't just a script; it's a modular framework.

* **Initialization:** **Xavier/Glorot Initialization** to prevent vanishing/exploding gradients in deep architectures.
* **Optimization:** Stochastic Gradient Descent (SGD) with **Momentum** and **Adaptive Learning Rate** decay.
* **Regularization:**
    * **Dropout:** Randomized neuron deactivation during training ($P_{keep} = 0.8$) to force redundant feature learning.
    * **L2 Regularization:** Weight decay penalization in the cost function.
* **Compression:** A custom **Pruning Algorithm** was developed.
    * *Result:* **92.5%** of network connections were pruned (weights $< |0.2|$) with only a **~3% drop in accuracy**, demonstrating the high redundancy of biological signal processing networks.

---

## 📊 Performance & Results

The model was validated using the **MIT-BIH Arrhythmia Database** (PhysioNet).

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Validation Accuracy** | **97.69%** | On synthetic dataset with Gaussian Noise |
| **Real Patient Accuracy** | **~95.2%** | Best case (Patient 19093) |
| **Generalization** | **>80%** | Across high-variance patient data |

*(Note: Real-world ECG signals were pre-processed (denoising/resampling) before inference.)*

---

## 📂 Repository Structure

The code follows a modular "Functional Programming" approach typical in Signal Processing research:

```bash
├── main.m                  # Entry point: Setup, Training, Evaluation
├── Entrena_DNN.m           # Training Loop (Epochs & Batching)
├── modules/
│   ├── forwardPropagation.m
│   ├── backPropagation.m   # Core Calculus Implementation
│   ├── computeCost.m       # BCE Loss
│   └── updateParams.m      # SGD/Adam Optimizer
├── utils/
│   ├── pruning.m           # Model Compression Logic
│   └── data_loader.m       # PhysioNet Parser
└── data/                   # MIT-BIH Samples (.mat)
