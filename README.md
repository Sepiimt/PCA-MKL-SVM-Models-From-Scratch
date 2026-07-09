
---
# **🔭 Project Overview**

This repository features a robust, from-scratch implementation of a **Multiple Kernel Learning Support Vector Machine (MKL-SVM)** coupled with a custom **Principal Component Analysis (PCA)** pipeline, built entirely in pure NumPy.

While high-level libraries like Scikit-Learn provide highly optimized single-kernel SVMs, they operate as rigid black boxes that require static, user-supplied kernel weights. This project breaks that limitation by building a dynamic architecture that actively _trains and optimizes_ kernel weights ($\beta$) using gradient descent during the learning process. It is built for complex, heterogeneous datasets where a single kernel function is insufficient, showcasing advanced quadratic programming, algorithmic optimization, and deep architectural transparency.

---
# **⏱️ TL;DR (Too Long; Didn't Read)**

If you only have 30 seconds, here is what makes this repository a significant engineering flex:
	
- **⚡Dynamic Weight Optimization:** Unlike Scikit-Learn (which only accepts static, user-defined kernel parameters), this model features a custom dual-optimization architecture that actively _trains and calibrates_ its own kernel weights ($\beta$) via gradient descent.
    
- **🚀Pure NumPy Engine:** Zero external machine learning libraries. The entire quadratic programming solver, Platt's search heuristics, and KKT boundary monitoring loops were written completely from scratch in vectorized NumPy.
    
- **💪The Big Flex (37 vs. 67):** The model achieves absolute mathematical parity with industry-standard frameworks ($97.3\%$ accuracy) but constructs its decision boundary using **nearly half the support vectors** (37 vs. 67). It matches the performance of standard libraries while maintaining a leaner, more efficient margin.
    
- **🏁High-Octane Throughput:** Handles a deeply nested optimization cap (1,200 outer $\beta$ loops $\times$ 800 inner SMO iterations) with extreme efficiency, averaging just $\sim0.21$ seconds per core iteration.

---
# **🛠️ Technical Highlights**

### _🌀 PCA Dimensionality reduction involvement_
Involvement of PCA dimensionality reduction model in the pipeline drastically aids the model's performance throughout the training and testing process, while maintaining the original information of the dataset up to ~ 0.961366 by reducing the data's dimensions from 29 to 10!

### _💡 Dynamic $\beta$ Training vs. Static Weights_
The defining achievement of this engine is its dual-optimization approach. Standard libraries do not train $\beta$ weights for multiple kernels. This model introduces an outer optimization loop that actively learns the optimal contribution of each base kernel.
	
- **🔎Proof of Convergence Stability:** When intentionally initialized with a skewed weight distribution (e.g., $\beta = [0.1, 0.2, 0.5, 0.2]$), the gradient descent step systematically corrects the bias. Through rigorous iteration, it naturally converges to a stable, isotropic balance (e.g., $\beta = [0.25, 0.25, 0.25, 0.25]$, or different values based on the provided data), proving the mathematical stability of the objective function.

### _🔥 High-Octane Sequential Minimal Optimization (SMO)_
The inner quadratic programming problem is solved without relying on external C-libraries (like `libsvm`). The custom `SMO` and `MKL` engine is heavily optimized for speed and intelligent traversal:
	
- **🧮Platt's Heuristic:** Instead of random pair selection, the SMO solver intelligently selects the second multiplier ($j$) by maximizing the step size $|E_i - E_j|$ (`sub_fit_j_vector_creator`), drastically accelerating convergence.
    
- **📸Granular KKT Monitoring:** The engine actively computes "Karush-Kuhn-Tucker" (KKT) boundary violations, allowing the loop to skip invalid pairs and focus computational cycles strictly on vectors that define the margin.
    
- **⏳Deep Computation Limits:** Nesting `max_SMO_iter = 800` inside a $\beta$-optimization loop of `max_beta_optimization_iter = 1200` results in millions of potential margin calculations. A standard **~4-minute training time** for this depth reflects highly optimized, vectorized NumPy array operations averaging just ~0.21 seconds per total `SMO` iteration of `800` rounds (or until convergence).

### _⚖️ Native Kernel Normalization_
When blending multiple kernels (e.g., Polynomial and RBF), massive variance in matrix magnitudes can destabilize the $\beta$ optimization. The `Kernels` metaclass dynamically enforces mathematical uniformity by scaling the kernel matrix using its diagonal elements:

$$K_{ij} \leftarrow \frac{K_{ij}}{\sqrt{K_{ii} K_{jj}}}$$

This guarantees that no single kernel mathematically dominates the others simply due to its native feature scale.

---
# **🧩 Data Preparation & Strict PCA Scaling**

Distance-based classifiers and variance-dependent algorithms are mathematically unforgiving when fed raw data.
	
- **💾Mandatory Data Standardization:** Principal Component Analysis is acutely sensitive to feature variance. Attempting to extract eigenvectors from unscaled data is a critical procedural error that skews the principal components toward arbitrary magnitudes. The `PCA` module strictly enforces zero-mean centering and unit-variance scaling (`scaler()`) before the covariance matrix is ever computed.
    
- **🔑Label Encoding:** Target vectors for the SVM are explicitly validated and mapped to $\{-1, 1\}$ to maintain the mathematical integrity of the margin calculations and KKT conditions.

---
# **🧠 Core Concepts**

### 1. ↕️ Principal Component Analysis (PCA)
High-dimensional data often contains noise and redundant features that exponentially increase the computational load of the `SMO` loop. The custom `PCA` implementation calculates the covariance matrix and extracts the top $k$ eigenvectors (via `np.linalg.eigh`) to project the data into a lower-dimensional subspace. This maximizes preserved variance while vastly reducing the computational footprint.

### 2. 🔀 Multiple Kernel Learning (MKL)
Real-world data is heterogeneous; a single kernel might underfit one aspect of the data while overfitting another. MKL evaluates the data through a linear combination of several base kernels:

$$K_{mkl}(x, z) = \sum_{k=1}^{K} \beta_k K_k(x, z)$$

Instead of forcing the engineer to guess the correct geometric transformation, this algorithm mathematically determines which structural perspectives best separate the classes.

---
# **⚙️ Logic & Process: The Dual Optimization**

Training this `MKL-SVM` requires solving two interwoven optimization problems:
	
1. **🔁The Inner Loop (SMO):** Maximizing the standard SVM dual formulation with respect to the Lagrange multipliers ($\alpha$), subject to the constraints $0 \leq \alpha_i \leq C$ and $\sum \alpha_i y^{(i)} = 0$.
    
2. **🔂The Outer Loop (Gradient Descent):** Calculating the gradient of the objective function with respect to each $\beta_k$, updating the weights iteratively, and projecting them back onto a simplex (ensuring $\sum \beta_k = 1$ and $\beta_k \geq 0$).

This relentless back-and-forth between $\alpha$ and $\beta$ optimization is what allows the model to correct skewed initializations and pinpoint the true global minimum.

---
# **🔬 Result Examination and Validation**

To validate the mathematical integrity of this custom Multiple Kernel Learning (MKL) implementation, the model was benchmarked against Scikit-Learn's industry-standard SVM.

### 🚧 The Baseline Limitation: Scikit-Learn's Static Architecture
It is critical to note that Scikit-Learn **does not natively support Multiple Kernel Learning**. It cannot actively train or optimize kernel weights ($\beta$) during execution. To create a valid 1-to-1 comparison, the custom engine was first allowed to train and organically discover the optimal kernel distribution. Once the custom model converged on an isotropic weight array of `[0.25, 0.25, 0.25, 0.25]`, these specific $\beta$ values were hardcoded and passed into Scikit-Learn's precomputed kernel matrix to establish the baseline.

### 🔧 Training Configuration
The custom model achieved convergence using the following hyperparameter framework:
	
- 🔂**$\beta$ Optimization Loop:** `beta_learning_rate = 0.1`, `max_iter = 1200`, `tolerance = 1e-3`
    
- **🔁**SMO Optimization Loop:** $C = 3$, `max_iter = 800`, `tolerance = 1e-3`

### 🚀 Performance Comparison

|**Metric**|**Custom MKL-SVM**|**Scikit-Learn (Injected β)**|
|---|---|---|
|**Accuracy**|0.9734|0.9734|
|**Precision**|0.9843|0.9843|
|**Recall**|0.9692|0.9692|
|**F1 Score**|0.9767|0.9767|
|**Kernel Target Alignment**|0.6469|0.6469|
|**Support Vector Count**|**37**|67|

### 🦺 Engineering Analysis
The cross-examination yields two major validations of the custom architecture:
	
1. **💯Absolute Mathematical Parity:** The identical scores in Accuracy, Precision, Recall, F1, and Kernel Target Alignment prove that the custom gradient descent and margin boundaries perfectly align with highly optimized, C-backed industry standards. The math is provably sound.
    
2. **👑Superior Margin Efficiency:** While both models achieved the exact same predictive performance, the custom SMO engine constructed its decision boundary using significantly fewer Support Vectors (37 versus Scikit-Learn's 67). This demonstrates that the custom KKT condition monitoring and sub-selection heuristics converged on a leaner, more tightly regularized hyperplane. By dropping redundant vectors, the custom model drastically reduces the computational footprint required for inference.
---
# **📂 Project Architecture**

| **File**         | **Role**              | **Description**                                                                                                                     |
| ---------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **`svm.py`**     | **The Interface**     | Handles kernel selection, argument parsing, and input validation before delegating to the core MKL engine.                          |
| `mkl.py`         | **The Orchestrator**  | Performs $\beta$ training and the orchestration of value parsing, fitting Kernels and SMO.                                          |
| **`smo.py`**     | **The Solver**        | Houses the custom Sequential Minimal Optimization logic, LIBSVM KKT heuristics, and $\alpha$-vector bounds clipping.                |
| **`kernels.py`** | **The Mathematician** | Vectorized NumPy implementations of base kernels (Linear, Polynomial, RBF, Sigmoid) featuring automatic cross-matrix normalization. |
| **`pca.py`**     | **The Filter**        | Dimensionality reduction engine handling strict data standardizations, covariance generation, and eigendecomposition.               |

---
# **🚀 Usage**

```python
import numpy as np
from pca import PCA
from svm import SVM

# --- 1. Dimensionality Reduction (Enforcing strict data scaling) ---
pca = PCA()
X_train_scaled = pca.scaler(X_train)
pca.fit(X_train_scaled)
X_train_reduced = pca.transform(X_train_scaled, n_components=10)

# --- 2. Initialize SVM and Select Kernels ---
# Selection Array: [Linear, Polynomial, RBF, Sigmoid]
model = SVM()
model.select_kernels([False, True, True, True]) 

# -- 3. Set Kernel Arguments (Set default_values=False for manual input) ---
model.set_kernels_arguments(X_train_reduced, default_values=True)

# --- 4. Train the Model (~4 minutes execution time for deep dual-optimization) ---
model.fit(
    X_train_reduced, y_train,
    smo_C=1.0, 
    max_SMO_iter=800,
    max_beta_optimization_iter=1200,
    detailed_timer=True
)

# --- 5. Extract Predictions ---
predictions = model.predict(X_test_reduced, strict_binary_result=True)
```

---
# **📄 License & Attribution**
	
- **Author:** Sepanta Metanat ([@Sepiimt](https://github.com/Sepiimt))
    
- **License:** This project is licensed under the "GPL-3.0 License" - see the [LICENSE](https://github.com/Sepiimt/PCA-MKL-SVM-Models-From-Scratch?tab=GPL-3.0-1-ov-file#) file for details.

---
