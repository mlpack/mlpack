# Neural Networks (ANN) - CLI Guide

`mlpack_ann` is a flexible command-line program for training deep neural networks. It supports classification, regression, and custom architectures defined via simple text files.

## 🚀 Quick Start

**1. Define your network:**
Create `model.txt`:
```
Linear 50
ReLU
Linear 10
LogSoftmax
```

**2. Train on your data:**
```bash
mlpack_ann -t train.csv -l labels.csv -L model.txt -M model.bin -v
```

**3. Predict:**
```bash
mlpack_ann -m model.bin -T test.csv -p predictions.csv -v
```

---

## 🏗️ Building a Network

Neural networks in mlpack are defined by stacking layers in a text file (one layer per line).

### Example Architecture
For a standard classification problem (e.g., MNIST):

```mermaid
graph TD;
    Input[Input Data (784 features)] --> L1[Linear 128];
    L1 --> A1[ReLU];
    A1 --> L2[Linear 64];
    L2 --> A2[ReLU];
    A2 --> Out[Linear 10];
    Out --> Act[LogSoftmax];
    Act --> Loss[NegativeLogLikelihood];
```

**File `mnist_arch.txt`:**
```
Linear 128      # Hidden layer 1
ReLU            # Activation
Linear 64       # Hidden layer 2
ReLU            # Activation
Linear 10       # Output layer (10 classes)
LogSoftmax      # Required for classification
```

---

## 📚 Supported Layers

### Core Layers
| Layer | Description | Example |
|-------|-------------|---------|
| `Linear <size>` | Fully connected layer with `size` neurons. | `Linear 128` |
| `Dropout <p>` | Randomly zeros input elements with probability `p`. | `Dropout 0.5` |
| `BatchNorm` | Batch Normalization. | `BatchNorm` |
| `Identity` | Pass-through layer (no operation). | `Identity` |

### Activation Functions
| Function | Description |
|----------|-------------|
| `ReLU` | Rectified Linear Unit ($f(x) = \max(0, x)$). |
| `LeakyReLU [alpha]` | Leaky ReLU with slope `alpha` (default 0.01). |
| `ELU [alpha]` | Exponential Linear Unit (default 1.0). |
| `Sigmoid` | Logistic function ($1 / (1 + e^{-x})$). |
| `Tanh` | Hyperbolic Tangent. |
| `HardTanH` | Faster approximation of Tanh. |
| `PReLU [alpha]` | Parametric ReLU (learnable slope). |
| `ReLU6` | ReLU capped at 6. |

### Output Layers
| Layer | Use Case |
|-------|----------|
| `LogSoftmax` | **Multi-class Classification**. Converts outputs to log-probabilities. |
| `Softmax` | Probability distribution (not recommended for NLL loss). |
| `(None)` | **Regression**. Use no activation or `Identity` for raw output. |

---

## 🛠️ Command-Line Reference

### Data Files
- `-t, --training_file`: **Required**. CSV file with training data (features).
- `-l, --labels_file`: **Required**. CSV file with training labels.
- `-T, --test_file`: CSV file with test data for prediction.
- `-L, --layers_file`: Text file defining network architecture.

### Model Management
- `-M, --output_model_file`: Save trained model to this file (e.g., `model.bin`).
- `-m, --input_model_file`: Load pre-trained model from this file.
- `-p, --predictions_file`: Save predictions to this CSV file.

### Training Parameters
- `-o, --optimizer`: Optimization algorithm. Options: `sgd`, `adam`, `rmsprop`. (Default: `rmsprop`)
- `-s, --step_size`: Learning rate. (Default: `0.01`)
- `-n, --max_iterations`: Number of training epochs/iterations. (Default: `1000`)
- `-b, --batch_size`: Size of mini-batches. (Default: `32`)
- `-r, --regression`: **Flag**. Enable regression mode (uses MSE loss). Default is Classification (NLL loss).

### Utilities
- `-v, --verbose`: Print detailed progress and info.
- `-S, --seed`: Set random seed for reproducibility.

---

## 💡 Examples

### 1. Classification (Iris Dataset)
Assuming `iris_train.csv` (4 features) and `iris_labels.csv` (3 classes: 0, 1, 2).

**Architecture `iris.txt`:**
```
Linear 10
ReLU
Linear 3
LogSoftmax
```

**Command:**
```bash
mlpack_ann -t iris_train.csv -l iris_labels.csv -L iris.txt \
           -o adam -s 0.001 -n 500 \
           -M iris_model.bin -v
```

### 2. Regression (Housing Prices)
Predicting a continuous value.

**Architecture `housing.txt`:**
```
Linear 64
ReLU
Linear 1
# No activation at the end for regression!
```

**Command:**
```bash
mlpack_ann -t housing.csv -l prices.csv -L housing.txt \
           --regression \
           -o rmsprop -s 0.01 \
           -p predictions.csv -v
```

### 3. Using a Pre-trained Model
Load `model.bin` and predict on new data `new_data.csv`:

```bash
mlpack_ann -m model.bin -T new_data.csv -p predictions.csv -v
```
