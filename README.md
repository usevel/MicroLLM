# 🧠 MicroLLM-CPP

![Language](https://img.shields.io/badge/Language-C%2B%2B17%20%2F%20C%2B%2B20-blue.svg?style=flat-square)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Pure%20STL)-success.svg?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Autoregressive%20MLP-purple.svg?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)

A minimalist, standalone **autoregressive Neural Network (MLP) engine** written completely from scratch in **pure modern C++** without any external machine learning or linear algebra libraries 

The project implements the fundamental mathematical mechanics of language modeling: custom tokenization, matrix forward propagation, manual gradient backpropagation, temperature-scaled sampling, and raw binary serialization.

---

## 🏗️ Architecture & Pipeline

```
[ Raw Training Corpus ]
         │
         ▼
[ Custom Tokenizer & Vocabulary Builder ]  ───► builds WordToIndex mapping
         │
         ▼
[ Context Sliding Window (N-gram) ]       ───► [W₁, W₂, W₃] ──► Target: [W₄]
         │
         ▼
[ Neural Network Forward Pass ]
     ├── Input Layer: Concatenated 1-Hot Vectors (ContextSize × VocabSize)
     ├── Hidden Layer: Linear Transform (W₁ · X + B₁) + ReLU Activation
     └── Output Layer: Linear Logits (W₂ · H + B₂)
         │
         ▼
[ Optimization (Backpropagation & SGD) ]
     ├── Output Error: E_out = (y_pred - y_target)
     ├── Hidden Error: E_hid = (W₂ᵀ · E_out) ⊙ ReLU'(H_raw)
     └── Weight Updates: W ← W - (η · E · Xᵀ)
         │
         ▼
[ Autoregressive Generation Engine ]
     ├── Numerically Stable Softmax: exp((zᵢ - max(z)) / T) / Σ
     ├── Weighted Roulette Sampling: std::mt19937
     └── Rolling Context Update: [W₂, W₃, W_next] ──► Next Iteration
```

---

## ✨ Key Features

### 1. Zero External Dependencies
* Written using standard C++ libraries only (`<vector>`, `<string>`, `<fstream>`, `<random>`, `<cmath>`).
* Highly portable — compiles on Windows (MSVC, MinGW), Linux (GCC, Clang), and macOS.

### 2. Manual Neural Network Math
* **Forward Propagation:** Direct matrix-vector multiplication with bias offsets.
* **Non-Linear Activation:** Standard **ReLU** ($f(x) = \max(0, x)$).
* **Backpropagation & SGD:** Explicit analytical derivative calculation and gradient propagation through all layers.

### 3. Text Generation with Temperature Sampling
* **Numerically Stable Softmax:** Mitigates floating-point overflow by shifting logits by the maximum value before exponentiation.
* **Temperature Parameter ($T$):** Configurable exploration control (lower values produce deterministic text, higher values increase creativity).
* **Stochastic Roulette Wheel:** Non-uniform random token selection based on cumulative probability mass.

### 4. Raw Binary Model Serialization
* Saves and restores model parameters (matrices $W_1, W_2$ and bias vectors $B_1, B_2$) directly to/from binary dumps (`brain_base.bin`) via raw byte streams.

---

## 🛠️ Mathematical Details

### Softmax with Temperature
Given output logits $z$, the probability of the $i$-th word is computed as:
$$P(w_i) = \frac{\exp\left(\frac{z_i - \max(z)}{T}\right)}{\sum_{j} \exp\left(\frac{z_j - \max(z)}{T}\right)}$$

### Backpropagation Gradients
* **Output Layer Error:** $\delta_{out} = \hat{y} - y$
* **Hidden Layer Error:** $\delta_{hid} = (W_2^T \delta_{out}) \cdot \mathbb{I}(h_{raw} > 0)$
* **Parameter Updates:** 
  $$\Delta W_2 = \eta (\delta_{out} \cdot h^T), \quad \Delta B_2 = \eta \delta_{out}$$
  $$\Delta W_1 = \eta (\delta_{hid} \cdot x^T), \quad \Delta B_1 = \eta \delta_{hid}$$

---

## 🚀 How to Build & Run

### Prerequisites
* C++17 or C++20 compatible compiler (MSVC 2019+, GCC 9+, Clang 10+)

### Building from Source
```bash
# Clone the repository
git clone https://github.com/usevel/MicroLLM.git
cd MicroLLM

# Compile with MSVC (Developer Command Prompt)
cl /EHsc /O2 /std:c++20 main.cpp /Fe:MicroLLM.exe

# Or compile with GCC / Clang
g++ -O3 -std=c++20 main.cpp -o MicroLLM
```

---

## 🎮 Interactive CLI Demo

```text
Создаем словарь
Перетащи файл: text.txt
-----------------------------------

Переобучить модель(+/-): +
Этап 1: запущено обучение 3000 эпох
файл сохранен brain_base.bin

Начните диалог: бармен налил еще
AI вывод: кофе и спросил тяжелый день на работе программист ответил очень ...
```

---

## 📜 License
Distributed under the **MIT License**. Feel free to inspect, modify, and use this code for learning and research.