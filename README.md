# CNN Implementation from Scratch

This repository demonstrates a complete implementation of a Convolutional Neural Network (CNN) from the ground up, without relying on high-level deep learning frameworks (such as TensorFlow or PyTorch). The goal is to build a CNN that classifies handwritten digits from the MNIST dataset using only NumPy for numerical operations and Matplotlib for visualizations.

### Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Key Features](#key-features)  
4. [Prerequisites & Dependencies](#prerequisites--dependencies)  
5. [Dataset Preparation (Windows)](#dataset-preparation-windows)  
6. [How to Run](#how-to-run)  
7. [Notebook Walkthrough](#notebook-walkthrough)  
   - [1. Imports](#1-imports)  
   - [2. Helper Functions](#2-helper-functions)  
   - [3. Data Loading & Preprocessing](#3-data-loading--preprocessing)  
   - [4. Model Architecture & Forward/Backward Pass](#4-model-architecture--forwardbackward-pass)  
   - [5. Training Loop](#5-training-loop)  
   - [6. Loss & Accuracy Visualization](#6-loss--accuracy-visualization)  
   - [7. Model Evaluation](#7-model-evaluation)  
8. [Hyperparameters & Configuration](#hyperparameters--configuration)  
9. [Results & Expected Performance](#results--expected-performance)  
10. [Troubleshooting & FAQs](#troubleshooting--faqs)  
11. [License](#license)  

---

## Project Overview

This project builds a CNN “from scratch” to classify the MNIST handwritten digits dataset (0–9). This is achieved by manually writing functions for:

- 2D convolution (forward and backward passes)  
- Max pooling (forward and backward passes)  
- ReLU activation (forward and backward passes)  
- Fully connected (dense) layer (forward and backward passes)  
- Softmax + Cross-Entropy loss (forward and backward)  
- End-to-end training loop using mini-batch gradient descent  

By the end of this notebook, you will have:  
1. A detailed understanding of how a CNN processes image data.  
2. Exposure to writing backpropagation logic for convolutional and pooling layers.  
3. Visualization of training loss and accuracy curves.  
4. A trained CNN model that achieves competitive accuracy on MNIST without any external DL library.  

---

## Repository Structure

```
/
├── CNN from scratch.ipynb     # Jupyter Notebook with full implementation
└── README.md                  # This detailed readme
```

> **Note:** On Windows, you do not need a separate `data\` folder. See the [Dataset Preparation](#dataset-preparation-windows) section below for instructions on where to place MNIST CSV files.

---

## Key Features

1. **No Deep-Learning Frameworks:** Implements every layer (convolution, pooling, activation, fully connected) using only NumPy.  
2. **Custom Forward & Backward Pass:** Every mathematical operation (convolution, pooling, ReLU, softmax) is derived and coded by hand.  
3. **Mini-Batch Gradient Descent:** Trains the network in mini-batches (batch size configurable).  
4. **Visualization:** Plots training loss and accuracy across epochs.  
5. **Modular Code:** Each subroutine (e.g., `conv2d`, `maxpool2d`, `relu`, etc.) is self-contained, making it easy to extend/modify.  
6. **Test Accuracy Evaluation:** After training, the notebook computes test-set accuracy and prints it.  

---

## Prerequisites & Dependencies

The notebook was developed and tested with **Python 3.12.7**. You will need the following Python packages:

- `numpy` – Numerical computations  
- `pandas` – Data loading (MNIST CSV)  
- `matplotlib` – Plotting loss/accuracy curves

Install these packages via pip:

```bash
pip install numpy pandas matplotlib
```

---

## Dataset Preparation (Windows)

On Windows, you do not need to create a separate `data\` folder. Simply download the MNIST CSV files and place them in the same directory as `CNN from scratch.ipynb`, or in any folder you prefer—just update the paths accordingly in the notebook.

### 1. Download MNIST CSV Files Manually

1. Open your web browser and go to:  
   [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
2. Click **Download** (it should download a ZIP file, likely named `mnist-in-csv.zip`).  
3. Once the ZIP is downloaded, right-click it and choose **“Extract All…”**.  
4. In the extraction wizard, you can choose to extract directly into the same folder as your notebook (so that you see `mnist_train.csv` and `mnist_test.csv` next to `CNN from scratch.ipynb`), or extract into a subfolder like `.\mnist-csv\`. The important thing is to note where those two CSVs end up.

### 2. Place CSV Files in Your Working Directory

- If you extracted directly into the folder containing `CNN from scratch.ipynb`, ensure you see:  
  ```
  <your-project-folder>  ├── CNN from scratch.ipynb
  ├── mnist_train.csv
  └── mnist_test.csv
  ```
- If you prefer a subfolder, e.g., `.\mnist_data\`, then your layout might be:  
  ```
  <your-project-folder>  ├── CNN from scratch.ipynb
  └── mnist_data      ├── mnist_train.csv
      └── mnist_test.csv
  ```
  In that case, you’ll need to adjust file paths in the notebook (e.g., use `pd.read_csv('mnist_data\mnist_train.csv')`).

> **Tip:** If you have the Kaggle CLI set up on Windows (requires `kaggle.json` in `%USERPROFILE%\.kaggle\`), you can also run:  
> ```powershell
> kaggle datasets download -d oddrationale/mnist-in-csv
> ```
> …then unzip the contents into the same folder as the notebook (or into a folder of your choice). This step is optional; manual download via the browser works fine.

---

## How to Run

1. **Ensure Dependencies Are Installed**  
   Open a Command Prompt or PowerShell window, then run:  
   ```powershell
   pip install numpy pandas matplotlib
   ```

2. **Download & Place MNIST Data**  
   Follow the [Dataset Preparation (Windows)](#dataset-preparation-windows) steps above to fetch `mnist_train.csv` and `mnist_test.csv`, and place them either alongside `CNN from scratch.ipynb` or in a subfolder of your choice.

3. **Adjust Paths (if needed)**  
   - By default, the notebook expects:  
     ```python
     pd.read_csv('mnist_train.csv')
     pd.read_csv('mnist_test.csv')
     ```  
     …in the same directory as the notebook.  
   - If you put the CSVs in a subfolder named `mnist_data`, update these lines to:  
     ```python
     pd.read_csv('mnist_data\\mnist_train.csv')
     pd.read_csv('mnist_data\\mnist_test.csv')
     ```

4. **Open & Run the Notebook**  
   - Double-click `CNN from scratch.ipynb` or open it via Jupyter Notebook in your browser:  
     ```powershell
     jupyter notebook
     ```  
   - Run all cells in order. The notebook will train the CNN, showing loss/accuracy after each epoch and final test accuracy.

---

## Notebook Walkthrough

Below is a high-level overview of each major section in `CNN from scratch.ipynb`. Use this as a guide to understand the code organization.

### 1. Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
- **pandas**: Load MNIST CSV files into DataFrames.  
- **NumPy**: Core numerical operations (matrix multiplications, convolutions, etc.).  
- **Matplotlib**: Plotting training curves.  

---

### 2. Helper Functions

Defines low-level routines used throughout the CNN:

1. **`conv2d`** (forward & backward)  
   - Implements 2D convolution via an im2col approach for efficiency.  
   - **Forward**: Pads input, constructs im2col matrix, multiplies with reshaped filters, then reshapes back to a feature map. Returns both the output activation and an im2col cache for backprop.  
   - **Backward**: Uses the cached columns to compute gradients w.r.t. \(W\), \(b\), and input \(X\) (removing padding when returning \(dX\)).

2. **`maxpool2d`** (forward & backward)  
   - **Forward**: Converts each \(2×2\) region into columns, takes the maximum to reduce spatial dimensions. Returns pooled output + max_indices cache.  
   - **Backward**: Propagates upstream gradients back only to the positions that were maxima during the forward pass (using the stored indices).

3. **`relu` / `relu_backward`**  
   - **Forward**: Computes \(\max(0, x)\).  
   - **Backward**: Zeros out gradients where \(x ≤ 0\).

4. **`fully_connected_forward` / `fully_connected_backward`**  
   - **Forward**: Computes \(Z = XW + b\).  
   - **Backward**: Given \(dZ\), computes \(dX = dZ W^T\), \(dW = X^T dZ\), and \(db = Σ dZ\).

5. **`softmax_cross_entropy_loss`**  
   - Converts raw scores \(Z\) into probabilities via softmax, computes cross-entropy loss, and returns both scalar loss and gradient \(dZ\).

---

### 3. Data Loading & Preprocessing

```python
# Adjust file paths if you placed CSVs in a subfolder.
train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')
train_data = df_train.values
test_data = df_test.values

np.random.shuffle(train_data)

Y_train = train_data[:, 0].astype(int)         
X_train = train_data[:, 1:] / 255.0 

Y_test = test_data[:, 0].astype(int)
X_test = test_data[:, 1:] / 255.0

X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

```

- **Normalization**: Dividing by 255 ensures pixel values lie in [0,1].  
- **Reshaping**: Converts each flat 784-element vector into a \(1×28×28\) tensor for convolution.

---

### 4. Model Architecture & Forward/Backward Pass

#### 4.1 Parameter Initialization

```python
def initialize_parameters():
    params = {}

    # Conv Layer 1: 8 filters, 1 channel, 3x3 kernel
    params['W1'] = np.random.rand(8,1,3,3) - 0.5
    params['b1'] = np.zeros((8,))

    # Conv Layer 2: 16 filters, 8 channels, 3x3 kernel
    params['W2'] = np.random.rand(16,8,3,3) - 0.5
    params['b2'] = np.zeros((16,))

    # FC Layer 1: 128 neurons
    params['W3'] = np.random.rand(128,16*7*7) - 0.5
    params['b3'] = np.zeros((128,))

    # FC Layer 2: 10 output classes
    params['W4'] = np.random.rand(10,128) - 0.5
    params['b4'] = np.zeros((10,))

    return params
```

- **Conv Layers**:  
  1. **Conv1**: Input=1 channel → Output=8 channels, \(3×3\), padding=1, stride=1  
  2. **Conv2**: Input=8 channels → Output=16 channels, \(3×3\), padding=1, stride=1  

- **Max-Pool Layers**:  
  - Each convolution is followed by a \(2×2\) max-pool with stride=2, halving spatial dimensions.  
    - After Conv1 & Pool: \(28×28 → 14×14\).  
    - After Conv2 & Pool: \(14×14 → 7×7\).  

- **Fully Connected Layers**:  
  1. **FC1**: \((16×7×7) → 128\)  
  2. **FC2**: \(128 → 10\) (number of classes)  

#### 4.2 Forward & Backward Functions

- **`conv2d(X, W, b, stride, padding)`**  
  - **Forward**: Applies zero-padding, rearranges input into columns (im2col), multiplies by flattened filters, reshapes back to activation volume. Returns both output activation and a cache (columns, shapes, etc.) needed for backprop.  
  - **Backward**: Uses cached columns to compute gradients w.r.t. \(W\), \(b\), and input \(X\) (removing padding when returning \(dX\)).

- **`maxpool2d(X, pool_size, stride)`**  
  - **Forward**: Rearranges each \(2×2\) region into columns, takes maximum along each column to downsample, and records indices of the maxima to a cache.  
  - **Backward**: Propagates upstream gradients back only to the positions that were maxima during the forward pass (using the stored indices).

- **`relu(X)` / `relu_backward(dA, cache)`**  
  - **Forward**: \(A = \max(0, X)\). Cache stores input \(X\).  
  - **Backward**: Zeros gradients where \(X \le 0\), else passes gradient through.

- **`fully_connected_forward(X, W, b)` / `fully_connected_backward(dZ, cache)`**  
  - **Forward**: Computes linear transform \(Z = XW + b\). Cache stores \((X, W, b)\).  
  - **Backward**: Given \(dZ\), computes \(dX = dZ W^T\), \(dW = X^T dZ\), and \(db = Σ dZ\).

- **`softmax_cross_entropy_loss(Z, Y_true)`**  
  - Converts raw scores \(Z\) into probabilities via softmax, computes cross-entropy loss, and returns both scalar loss and gradient \(dZ\).

---

### 5. Training Loop

```python
epochs = 10
alpha = 0.01
batch_size = 32

#Initialize the params
params = initialize_parameters()

losses = []
train_accuracy = []

N = X_train.shape[0]
#train loop
for epoch in range(epochs):
    # Shuffle data
    perm = np.random.permutation(N)
    X_train_shuffled = X_train[perm]
    Y_train_shuffled = Y_train[perm]

    epoch_loss = 0
    for i in range(0, N, batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        Y_batch = Y_train_shuffled[i:i+batch_size]

        # Forward and backward pass for the batch
        batch_loss, params = forward_backward(X_batch, Y_batch, params, alpha)
        epoch_loss += batch_loss
        
    
    # Predictions after epoch
    train_preds = predict(X_train, params)
    train_acc = np.mean(train_preds == Y_train.flatten()) * 100

    print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f} | Train Accuracy = {train_acc:.2f}%")
    losses.append(epoch_loss)
    train_accuracy.append(train_acc)
```

- **Data Shuffling**: Ensures each epoch sees data in a different order.  
- **Mini-Batch Gradient Descent**: Splits training set into batches of size `batch_size`.  
- **Metrics**:  
  - **Average training loss** per epoch.  
  - **Training accuracy** (percentage correct).  

---

### 6. Loss & Accuracy Visualization

After training completes, visualize trends:

```python
#plotting training loss and training accuracy
fig,axes = plt.subplots(2,1)
axes[0].plot(losses)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss over Epochs')
axes[0].legend()

axes[1].plot(train_accuracy)
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Training accuracy')
axes[1].set_title('Training Accuracy over Epochs')
axes[1].legend()

plt.tight_layout()
plt.show()
```

- **Interpretation**:  
  - A steadily decreasing loss curve combined with an increasing accuracy curve indicates effective learning.  
  - Monitor for signs of overfitting (e.g., accuracy plateaus or diverges).

---

### 7. Model Evaluation

After training, evaluate on the held-out test set:

```python
def test_accuracy(X_test, Y_test, params, batch_size=128):
    predictions = predict(X_test, params, batch_size=batch_size)
    acc = np.mean(predictions == Y_test) * 100
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# Example usage after training:
test_acc = test_accuracy(X_test, Y_test, trained_params)
```

- **`predict(X, params, batch_size)`**  
  - Iterates over `X` in batches, performs only the forward pass (conv → relu → pool → … → softmax), and returns predicted labels.  
- **Expected Test Accuracy on MNIST**:  
  - Typically **95%–98%**, depending on random initialization and hyperparameters.

---

## Hyperparameters & Configuration

All key hyperparameters are defined near the top or within function calls:

- **Number of Epochs** (`epochs`)  
  - Default: **10** (adjustable; more epochs often improve accuracy at the cost of longer runtime).  

- **Learning Rate** (`learning_rate`)  
  - Default: **0.01** (experiment with values like 0.005 or 0.001).  

- **Batch Size** (`batch_size`)  
  - Default: **64** (try 32 or 128, depending on your system’s RAM).  

- **Number of Filters**  
  - Conv1: **8** filters (3×3)  
  - Conv2: **16** filters (3×3)  

- **Hidden Layer Size**  
  - FC1: **128** units  

- **Weight Initialization**    
    - Conv layers:  
      ```python
      np.random.rand(num_filters, in_channels, fh, fw) - 0.5
      ```  
    - FC layers:  
      ```python
      np.random.rand(fan_in, fan_out) - 0.5
      ```  

To tune these:  
1. Modify values in the `initialize_params()` function.  
2. Adjust `learning_rate`, `epochs`, and `batch_size` when training.  

---

## Results & Expected Performance

- **Training Loss & Accuracy** (after 10 epochs with default hyperparameters):  
  - **Training Accuracy**: ~97%–99%   

- **Test Accuracy**:  
  - Typically **95%–98%** on MNIST, depending on random seed and hyperparameters.  

Feel free to:  
- Increase `epochs` to 20 or 30 to push test accuracy above 98%.  
- Experiment with additional convolutional layers or dropout to improve generalization.  
- Monitor for overfitting: if training accuracy → 100% but test accuracy stagnates, consider regularization or data augmentation.

---

## Troubleshooting & FAQs

1. **Notebook Fails on `pd.read_csv`**  
   - Ensure the CSV files are named exactly `mnist_train.csv` and `mnist_test.csv` (case-sensitive on some systems) and are in the same directory as the notebook—or adjust the paths accordingly.  
   - Verify that each CSV is a proper comma-separated file (open in Notepad or Excel to check).

2. **MemoryError / Running Out of RAM**  
   - MNIST CSVs require about 35–40 MB of RAM. On most modern Windows machines with ≥4 GB RAM, this should be fine.  
   - If memory is limited, reduce `batch_size` (e.g., to 32 or 16) or run on a machine with more RAM.

3. **Training Is Very Slow**  
   - Pure NumPy convolution/backprop can be slow compared to GPU-accelerated frameworks or optimized libraries.  
   - To speed up:  
     - Reduce the number of filters (e.g., Conv1→4 filters, Conv2→8 filters).  
     - Decrease `epochs` or increase `batch_size`.  
     - Ensure you’re not running other heavy processes simultaneously.

4. **Accuracy Stagnates or Doesn’t Improve**  
   - Lower the learning rate to 0.005 or 0.001.  
   - Verify that normalization (`X /= 255.0`) is correctly applied.  
   - Check that backward functions properly propagate gradients—print out shapes or intermediate values if unsure.

5. **Shape Mismatch or Broadcasting Errors**  
   - Print shapes after each major operation (`conv2d`, `maxpool2d`, flatten, fully connected) to confirm alignment.  
   - Ensure that pooling backward receives the correct cache (indices of max positions).

6. **Kaggle CLI Doesn’t Work on Windows or API Key Issues**  
   - If you prefer, simply download the dataset manually via the Kaggle website, as described above.  
   - If using the CLI on Windows, place `kaggle.json` in `%USERPROFILE%\.kaggle\kaggle.json` and set its file permissions so only you can read it:  
     1. In File Explorer, right-click `kaggle.json` → **Properties** → **Security** → Edit permissions.  
     2. Ensure your user account has **Full control**, and remove “Everyone” if it’s listed, or set their permissions to “Read” only.

---

## License

This project is released under the **MIT License**. Feel free to clone, modify, and use this code for educational and personal purposes. Attribution is appreciated.

---

**Author:**  
Pujit Shetty(@Quasar4606) 
— A hand-crafted CNN from scratch for MNIST classification —
