# Mathematical Foundations

This document details the mathematical formulations and derivations for Triune's optimization algorithms, routing mechanism, and runtime memory bounds.

---

## 1. Symmetrical Dual-Sided Centroid-Steered GaLore

### 1.1 SVD-Based Gradient Subspace Projections

Let $W \in \mathbb{R}^{m \times n}$ denote the parameter matrix of a linear projection layer, and $G = \nabla_W \mathcal{L} \in \mathbb{R}^{m \times n}$ denote the full-rank gradient of the loss $\mathcal{L}$ with respect to $W$.

In low-rank gradient projection (GaLore), the gradient $G$ is projected into a lower-rank subspace to reduce optimizer state memory. The projection direction is chosen based on matrix dimensions:

#### Case 1: Left-Side Projection ($m \ge n$)
For layers expanding the hidden dimension ($m \ge n$), projection occurs from the left:
$$P \in \mathbb{R}^{m \times r}, \quad P^T P = I_r \quad (r \ll \min(m, n))$$

The projected gradient $\tilde{G}_L \in \mathbb{R}^{r \times n}$ is:
$$\tilde{G}_L = P^T G$$

The update step $\Delta W \in \mathbb{R}^{m \times n}$ is reconstructed via:
$$\Delta W = P \tilde{G}_{L, \text{opt}}$$

#### Case 2: Right-Side Projection ($m < n$)
For layers contracting the hidden dimension ($m < n$), projection occurs from the right:
$$Q \in \mathbb{R}^{n \times r}, \quad Q^T Q = I_r \quad (r \ll \min(m, n))$$

The projected gradient $\tilde{G}_R \in \mathbb{R}^{m \times r}$ is:
$$\tilde{G}_R = G Q$$

The update step is reconstructed via:
$$\Delta W = \tilde{G}_{R, \text{opt}} Q^T$$

---

### 1.2 Centroid-Steered Subspace Augmentation

Let $\mathcal{C} = \{c_e\}_{e=1}^E$ denote activation centroids for expert $e$, where $c_e \in \mathbb{R}^d$ is the mean representation vector of tokens routed to expert $e$.

For weight matrix $W_e \in \mathbb{R}^{m \times n}$, let $c_{\text{projected}} \in \mathbb{R}^{d_{\text{proj}}}$ match the active projection dimension ($d_{\text{proj}} = m$ for left-projection, $d_{\text{proj}} = n$ for right-projection).

Let $\Phi \in \mathbb{R}^{d_{\text{proj}} \times r}$ denote the active projection basis ($P$ or $Q$). The orthogonal projection of $c_{\text{projected}}$ onto the column space of $\Phi$ is:
$$c_{\text{proj}} = \Phi \Phi^T c_{\text{projected}}$$

The residual component orthogonal to $\Phi$ is:
$$c_{\text{res}} = c_{\text{projected}} - c_{\text{proj}}$$

Normalizing yields an orthonormal steering vector:
$$c_{\text{orth}} = \frac{c_{\text{res}}}{\|c_{\text{res}}\|_2}, \quad \text{where } \Phi^T c_{\text{orth}} = \mathbf{0}_r$$

The projection subspace is augmented with $c_{\text{orth}}$ scaled by steering factor $\alpha \ge 0$:
$$\Phi_{\text{aug}} = \begin{bmatrix} \Phi & \alpha c_{\text{orth}} \end{bmatrix} \in \mathbb{R}^{d_{\text{proj}} \times (r + 1)}$$

---

### 1.3 Subspace Orthogonality

#### Theorem 1
Let $\Phi \in \mathbb{R}^{d_{\text{proj}} \times r}$ satisfy $\Phi^T \Phi = I_r$, and let $c_{\text{orth}} \in \mathbb{R}^{d_{\text{proj}}}$ satisfy $\|c_{\text{orth}}\|_2 = 1$ and $\Phi^T c_{\text{orth}} = \mathbf{0}_r$. Then the Gram matrix of $\Phi_{\text{aug}} = \begin{bmatrix} \Phi & \alpha c_{\text{orth}} \end{bmatrix}$ is block-diagonal:

$$\Phi_{\text{aug}}^T \Phi_{\text{aug}} = \begin{bmatrix} I_r & \mathbf{0}_r \\ \mathbf{0}_r^T & \alpha^2 \end{bmatrix}$$

#### Proof
$$\Phi_{\text{aug}}^T \Phi_{\text{aug}} = \begin{bmatrix} \Phi^T \\ \alpha c_{\text{orth}}^T \end{bmatrix} \begin{bmatrix} \Phi & \alpha c_{\text{orth}} \end{bmatrix} = \begin{bmatrix} \Phi^T \Phi & \alpha \Phi^T c_{\text{orth}} \\ \alpha c_{\text{orth}}^T \Phi & \alpha^2 c_{\text{orth}}^T c_{\text{orth}} \end{bmatrix}$$

Applying $\Phi^T \Phi = I_r$, $\Phi^T c_{\text{orth}} = \mathbf{0}_r$, and $c_{\text{orth}}^T c_{\text{orth}} = 1$:
$$\Phi_{\text{aug}}^T \Phi_{\text{aug}} = \begin{bmatrix} I_r & \mathbf{0}_r \\ \mathbf{0}_r^T & \alpha^2 \end{bmatrix}$$
$\blacksquare$

The original $r$ gradient coordinate axes and the $(r+1)$-th steering coordinate axis remain mutually orthogonal, preventing cross-interference in optimizer momentum and variance updates.

---

### 1.4 Optimizer Memory Complexity

For a Mixture-of-Experts down-projection layer $W \in \mathbb{R}^{D_{\text{model}} \times D_{\text{ffn}}}$ ($D_{\text{model}} = 1536$, $D_{\text{ffn}} = 9216$, rank $r = 256$):

- **Standard AdamW**:
  $$2 \times (D_{\text{model}} \times D_{\text{ffn}}) = 2 \times 1536 \times 9216 = 28,311,552 \text{ states}$$

- **Left-Side Projection ($P \in \mathbb{R}^{m \times r}$)**:
  $$2 \times (r \times D_{\text{ffn}}) = 2 \times 256 \times 9216 = 4,718,592 \text{ states}$$

- **Right-Side Symmetrical Projection ($Q \in \mathbb{R}^{n \times r}$)**:
  $$2 \times (D_{\text{model}} \times r) = 2 \times 1536 \times 256 = 786,432 \text{ states}$$

$$\frac{\text{States}_{\text{Left}}}{\text{States}_{\text{Right}}} = \frac{4,718,592}{786,432} = 6.0\times$$

Selecting right-side projection for contracting layers reduces optimizer state memory by $6\times$ compared to left-only projection.

---

## 2. Differentiable Straight-Through Gumbel-Softmax Routing

### 2.1 Categorical Reparameterization

Let $H_{\text{prefix}} \in \mathbb{R}^{B \times T \times D}$ denote the sequence representation from prefix layers. The sequence-pooled context $s \in \mathbb{R}^D$ is:
$$s = \frac{1}{T} \sum_{t=1}^T H_{\text{prefix}, t}$$

The router MLP projects $s$ to unnormalized exit logits $\pi \in \mathbb{R}^K$ ($K=3$: Reflex, Limbic, Cortex):
$$\pi = \mathbf{W}_2 \cdot \operatorname{ReLU}(\mathbf{W}_1 \cdot s + b_1) + b_2$$

With standard Gumbel noise $G_i = -\log(-\log(U_i)), U_i \sim \operatorname{Uniform}(0, 1)$, continuous relaxed routing probabilities are:
$$y_{\text{soft}, i} = \frac{\exp\left(\frac{\pi_i + G_i}{\tau}\right)}{\sum_{j=1}^K \exp\left(\frac{\pi_j + G_j}{\tau}\right)}$$
where $\tau > 0$ is the annealing temperature.

### 2.2 Straight-Through (ST) Estimator

To execute discrete pathways while maintaining gradient flow:
$$y_{\text{hard}} = \operatorname{one\_hot}\left(\operatorname{argmax}(y_{\text{soft}})\right)$$
$$y_{\text{ST}} = y_{\text{hard}} + y_{\text{soft}} - \operatorname{detach}(y_{\text{soft}})$$

- **Forward**: $y_{\text{ST}} = y_{\text{hard}}$ (discrete single-path execution).
- **Backward**: $\frac{\partial y_{\text{ST}}}{\partial \pi} = \frac{\partial y_{\text{soft}}}{\partial \pi}$ (continuous gradient flow to router weights).

---

## 3. Muon Optimizer (Newton-Schulz Orthogonalization)

Muon optimizes non-expert 2D weight projections by computing the polar factor of the gradient momentum using 5th-order Newton-Schulz iterations.

### 3.1 Polar Factor Recurrence

Let $G \in \mathbb{R}^{m \times n}$ ($m \ge n$) denote the unscaled momentum buffer. The initial normalized matrix is:
$$X_0 = \frac{G}{\|G\|_F + \epsilon} \cdot \frac{1}{\sqrt{\max(m, n)}}$$

At each step $k \in \{0, \dots, K-1\}$ (typically $K=5$):
$$B_k = X_k^T X_k \in \mathbb{R}^{n \times n}$$
$$X_{k+1} = X_k \left( a I_n + b B_k + c B_k^2 \right)$$

### 3.2 Optimal 5th-Order Coefficients

The coefficients are derived via Chebyshev minimax approximation of $f(s) = s^{-1/2}$ on $s \in (0, 1]$:
$$a = \frac{3446}{1024} \approx 3.365234375$$
$$b = -\frac{4765}{1024} \approx -4.6533203125$$
$$c = \frac{2263}{1024} \approx 2.2100000000$$

### 3.3 Memory Footprint

AdamW maintains both first moment $m_t$ and second moment $v_t$ (8 bytes per parameter in FP32). Muon tracks only first-moment momentum $m_t$ and computes orthogonal updates directly, reducing optimizer state storage by $2\times$.

---

## 4. Intermediate Exit Variance Normalization

Intermediate exit heads take activations from intermediate layers where representation scale $\operatorname{Var}(x_l)$ differs from the terminal layer $x_{\text{Cortex}}$:
$$\operatorname{Var}(x_l) \ne \operatorname{Var}(x_{\text{Cortex}})$$

Triune applies parameter-free `RMSNorm` prior to exit projections:
$$\tilde{x}_l = \operatorname{RMSNorm}(x_l) = \frac{x_l}{\sqrt{\frac{1}{D} \sum_{i=1}^D x_{l, i}^2 + \epsilon}}$$
$$\operatorname{logits}_l = W_{\text{exit}, l} \cdot \tilde{x}_l$$

This equalizes representation norms:
$$\|\tilde{x}_{\text{Reflex}}\|_2 \approx \|\tilde{x}_{\text{Limbic}}\|_2 \approx \|\tilde{x}_{\text{Cortex}}\|_2 = \sqrt{D}$$

guaranteeing balanced gradient scales across all exit heads in the total training loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}}(\text{selected}) + \lambda_r \mathcal{L}_{\text{router}} + \lambda_b \mathcal{L}_{\text{balance}}$$

---

## 5. Layer Streaming Memory Bounds

Under Triune's Layer Streaming engine:

1. Root parameters (embeddings, router prefix, exit heads) remain pinned in GPU memory ($V_{\text{root}}$).
2. Transformer layer parameters are stored in host RAM in `float8_e4m3fn`.
3. Only the current block $l$ and prefetching block $l+1$ occupy GPU memory.
4. Optimizer states for layer $l$ are transferred to GPU only during that layer's optimization step.

### Peak VRAM Bound

$$V_{\text{peak}} = V_{\text{root}} + \max_l \left( V_{\text{param}, l} + V_{\text{param}, l+1} + V_{\text{act}, l} + V_{\text{opt}, l} \right)$$

For `triune-2.5b` ($D=1280$, 8 experts, $B=1, T=128$):
$$V_{\text{root}} \approx 118\text{ MB}, \quad V_{\text{param}, l} \approx 65\text{ MB}, \quad V_{\text{act}, l} \approx 12\text{ MB}, \quad V_{\text{opt}, l} \approx 80\text{ MB}$$
$$V_{\text{peak}} \le 118 + 65 + 65 + 12 + 80 = 340\text{ MB}$$
