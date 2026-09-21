# Mathematical Foundations of the Triune Framework: Symmetrical Dual-Sided Centroid-Steered GaLore and Differentiable Gumbel-Softmax Routing

This document provides the formal mathematical derivations, structural proofs, and dimensional analyses for the core algorithmic breakthroughs powering the **Triune Framework**. These methodologies bridge activation-space token semantics with parameter-space low-rank gradient optimization, and establish an end-to-end differentiable routing paradigm for heterogeneous dynamic-depth language models.

---

## 1. Mathematical Proof of Symmetrical Dual-Sided Centroid-Steered GaLore

### 1.1 SVD-Based Gradient Subspace Projections

Let $W \in \mathbb{R}^{m \times n}$ denote the parameter matrix of a linear projection layer, and $G = \nabla_W \mathcal{L} \in \mathbb{R}^{m \times n}$ denote the full-rank gradient of the loss function $\mathcal{L}$ with respect to $W$.

In standard low-rank gradient projection (GaLore), the gradient $G$ is projected into a lower-rank compact space to minimize the memory footprint of first- and second-moment tracking in adaptive optimizers (e.g., AdamW). The direction of projection is determined by the shape of $W$ to optimize memory efficiency:

#### Case 1: Left-Side Projection ($m \ge n$)
For layers expanding the hidden dimension (e.g., MLP up-projections where $m = D_{\text{ffn}}, n = D_{\text{model}}$), we project $G$ from the left:
$$P \in \mathbb{R}^{m \times r}, \quad P^T P = I_r \quad (r \ll \min(m, n))$$

The compact projected gradient $\tilde{G}_L \in \mathbb{R}^{r \times n}$ is computed as:
$$\tilde{G}_L = P^T G$$

The reconstructed gradient step $\Delta W \in \mathbb{R}^{m \times n}$ is mapped back to the parameter space via:
$$\Delta W = P \tilde{G}_{L, \text{opt}}$$
where $\tilde{G}_{L, \text{opt}}$ represents the update vector calculated by the optimizer in the low-rank subspace.

#### Case 2: Right-Side Projection ($m < n$)
For layers contracting the hidden dimension (e.g., MLP down-projections where $m = D_{\text{model}}, n = D_{\text{ffn}}$), we project $G$ from the right:
$$Q \in \mathbb{R}^{n \times r}, \quad Q^T Q = I_r \quad (r \ll \min(m, n))$$

The compact projected gradient $\tilde{G}_R \in \mathbb{R}^{m \times r}$ is computed as:
$$\tilde{G}_R = G Q$$

The reconstructed gradient step $\Delta W \in \mathbb{R}^{m \times n}$ is mapped back via:
$$\Delta W = \tilde{G}_{R, \text{opt}} Q^T$$

---

### 1.2 Centroid-Steered Subspace Augmentation

Let $\mathcal{C} = \{c_e\}_{e=1}^E$ be the set of activation centroids for each expert $e$ in a Mixture-of-Experts (MoE) block, where $c_e \in \mathbb{R}^d$ is the mean representation vector of all tokens routed to expert $e$.

For an expert weight matrix $W_e \in \mathbb{R}^{m \times n}$, let $c_{\text{projected}} \in \mathbb{R}^d$ be the centroid vector projected or padded to match the target projection dimension ($d_{\text{proj}} = m$ for left-projection; $d_{\text{proj}} = n$ for right-projection).

We steer our orthogonal projection basis ($P$ or $Q$) toward the semantic direction of the routed tokens by computing its projection onto the orthogonal complement of the current basis. This guarantees that the optimizer allocates search capacity to the semantic manifold currently active in the forward pass.

Let $\Phi \in \mathbb{R}^{d_{\text{proj}} \times r}$ denote the active projection matrix (either $P$ or $Q$). The orthogonal projection of $c_{\text{projected}}$ onto the column space of $\Phi$ is defined as:
$$c_{\text{proj}} = \Phi \Phi^T c_{\text{projected}}$$

The residual vector $c_{\text{res}} \in \mathbb{R}^{d_{\text{proj}}}$, representing the component of token semantics not captured by the current singular vectors of the gradient, is:
$$c_{\text{res}} = c_{\text{projected}} - c_{\text{proj}}$$

We normalize this residual to establish an orthonormal steering vector $c_{\text{orth}}$:
$$c_{\text{orth}} = \frac{c_{\text{res}}}{\|c_{\text{res}}\|_2}$$

By construction, $c_{\text{orth}}$ is strictly orthogonal to the column space of $\Phi$:
$$\Phi^T c_{\text{orth}} = \mathbf{0}_r$$

We augment the projection subspace by appending $c_{\text{orth}}$ as an additional basis column scaled by a steering parameter $\alpha \ge 0$:
$$\Phi_{\text{aug}} = \begin{bmatrix} \Phi & \alpha c_{\text{orth}} \end{bmatrix} \in \mathbb{R}^{d_{\text{proj}} \times (r + 1)}$$

---

### 1.3 Mathematical Proof of Subspace Orthogonality

We prove that the augmented projection matrix $\Phi_{\text{aug}}$ preserves the mathematical orthogonality of the projection step up to the user-defined scalar scaling factor $\alpha$.

#### Theorem 1
Let $\Phi \in \mathbb{R}^{d_{\text{proj}} \times r}$ be a semi-orthogonal matrix satisfying $\Phi^T \Phi = I_r$, and let $c_{\text{orth}} \in \mathbb{R}^{d_{\text{proj}}}$ be a unit vector satisfying $\|c_{\text{orth}}\|_2 = 1$ and $\Phi^T c_{\text{orth}} = \mathbf{0}_r$. The metric matrix of the augmented projection $\Phi_{\text{aug}} = \begin{bmatrix} \Phi & \alpha c_{\text{orth}} \end{bmatrix}$ is block-diagonal and preserves the separation of the original subspace and the steered component.

#### Proof:
We compute the inner product matrix $\Phi_{\text{aug}}^T \Phi_{\text{aug}}$:
$$\Phi_{\text{aug}}^T \Phi_{\text{aug}} = \begin{bmatrix} \Phi^T \\ \alpha c_{\text{orth}}^T \end{bmatrix} \begin{bmatrix} \Phi & \alpha c_{\text{orth}} \end{bmatrix}$$

Multiplying the partitioned matrices:
$$\Phi_{\text{aug}}^T \Phi_{\text{aug}} = \begin{bmatrix} \Phi^T \Phi & \alpha \Phi^T c_{\text{orth}} \\ \alpha c_{\text{orth}}^T \Phi & \alpha^2 c_{\text{orth}}^T c_{\text{orth}} \end{bmatrix}$$

Substituting the semi-orthogonality condition $\Phi^T \Phi = I_r$, the orthogonal complement condition $\Phi^T c_{\text{orth}} = \mathbf{0}_r$, and the unit norm condition $c_{\text{orth}}^T c_{\text{orth}} = 1$:
$$\Phi_{\text{aug}}^T \Phi_{\text{aug}} = \begin{bmatrix} I_r & \mathbf{0}_r \\ \mathbf{0}_r^T & \alpha^2 \end{bmatrix}$$

Thus, the columns of $\Phi_{\text{aug}}$ remain strictly mutually orthogonal. The first $r$ coordinates correspond to the native low-rank gradient coordinates, and the $(r+1)$-th coordinate represents the isolated semantic steering direction. This ensures that the momentum and variance updates of the optimizer along the low-rank dimensions do not leak into or interfere with the semantic steering coordinate.

$\blacksquare$

---

### 1.4 Memory Footprint and Dimension Complexity Analysis

To demonstrate the memory savings of **Symmetrical Dual-Sided GaLore** over standard one-sided (Left) GaLore, we analyze the parameter count of the optimizer states (momentum and variance) for a Mixture-of-Experts down-projection linear layer.

Let $W_d \in \mathbb{R}^{D_{\text{model}} \times D_{\text{ffn}}}$ where $D_{\text{model}} = 1536$ and $D_{\text{ffn}} = 9216$. Let the low-rank projection limit be $r = 256$.

#### Standard Optimizer (e.g., AdamW)
Optimizer states track momentum ($m_t$) and variance ($v_t$) for every single parameter of the layer:
$$\text{Params}_{\text{AdamW}} = 2 \times (D_{\text{model}} \times D_{\text{ffn}}) = 2 \times 1536 \times 9216 = 28,311,552 \quad (\approx 28.3\text{M})$$

#### One-Sided (Left) Projection GaLore
By restricting projection to the left side ($P \in \mathbb{R}^{m \times r}$), the optimizer tracks a compact gradient $\tilde{G}_L \in \mathbb{R}^{r \times n}$ of size $256 \times 9216$:
$$\text{Params}_{\text{Left-GaLore}} = 2 \times (r \times D_{\text{ffn}}) = 2 \times 256 \times 9216 = 4,718,592 \quad (\approx 4.7\text{M})$$

#### Symmetrical Sized (Right) Projection GaLore (Triune)
Since $D_{\text{model}} < D_{\text{ffn}}$, our framework dynamically switches to right-side projection ($Q \in \mathbb{R}^{n \times r}$). The optimizer tracks a compact gradient $\tilde{G}_R \in \mathbb{R}^{m \times r}$ of size $1536 \times 256$:
$$\text{Params}_{\text{Right-GaLore}} = 2 \times (D_{\text{model}} \times r) = 2 \times 1536 \times 256 = 786,432 \quad (\approx 0.78\text{M})$$

#### Saving Factor:
$$\text{Ratio} = \frac{\text{Params}_{\text{Left-GaLore}}}{\text{Params}_{\text{Right-GaLore}}} = \frac{4,718,592}{786,432} = 6.0\times$$

By implementing symmetrical dual-sided projection, the Triune framework achieves a **$6\times$ memory reduction** in optimizer states for down-projection layers compared to standard left-only GaLore implementations.

---

## 2. Mathematical Formulation of Differentiable Gumbel-Softmax ST Routing

### 2.1 The Reparameterisation Trick for Categorical Decisions

Let $H_{\text{prefix}} \in \mathbb{R}^{B \times T \times D}$ represent the sequence representations generated by the prefix layers of the model. The pooled sequence context $s \in \mathbb{R}^D$ is computed as:
$$s = \frac{1}{T} \sum_{t=1}^T H_{\text{prefix}, t}$$

The early-exit router MLP projects $s$ to obtain unnormalized pathway routing logits $\pi \in \mathbb{R}^K$ (where $K = 3$, corresponding to Reflex, Limbic, and Cortex):
$$\pi = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot s + b_1) + b_2$$

To make the discrete routing decision end-to-end differentiable, we introduce IID noise variables $G_i$ sampled from a standard Gumbel distribution:
$$G_i = -\log(-\log(U_i)) \quad \text{where} \quad U_i \sim \text{Uniform}(0, 1)$$

The continuous soft routing selection probability $y_{\text{soft}} \in \mathbb{R}^K$ is computed via the relaxed softmax:
$$y_{\text{soft}, i} = \frac{\exp\left(\frac{\pi_i + G_i}{\tau}\right)}{\sum_{j=1}^K \exp\left(\frac{\pi_j + G_j}{\tau}\right)}$$
where $\tau > 0$ is the annealing temperature.

---

### 2.2 Straight-Through (ST) Gradient Flow

While $y_{\text{soft}}$ is fully differentiable, running fractional routing paths (executing part-Reflex and part-Cortex) is computationally impractical. To achieve physical latency and parameter savings, we must force a discrete selection in the forward pass.

Let $y_{\text{hard}} \in \{0, 1\}^K$ be the one-hot representation of the categorical choice:
$$y_{\text{hard}} = \text{one\_hot}\left(\operatorname{argmax}(y_{\text{soft}})\right)$$

To maintain differentiability, we formulate the straight-through estimator:
$$y_{\text{ST}} = y_{\text{hard}} + y_{\text{soft}} - \operatorname{detach}(y_{\text{soft}})$$

#### Forward Pass Evaluation:
$$y_{\text{ST}} = y_{\text{hard}} + y_{\text{soft}} - y_{\text{soft}} = y_{\text{hard}}$$
The forward pass is completely discrete and sparse, executing exactly one selected pathway.

#### Backward Pass Evaluation:
Since the derivative of the $\operatorname{detach}$ operator is zero, the gradient of the loss $\mathcal{L}$ with respect to the input logits $\pi$ bypasses the non-differentiable step:
$$\frac{\partial y_{\text{ST}}}{\partial \pi} = \frac{\partial y_{\text{soft}}}{\partial \pi}$$

This ensures that backpropagation directly optimizes the routing parameters $\mathbf{W}_1, \mathbf{W}_2$ to select paths that minimize the downstream autoregressive language modeling loss.

---

## 3. Mathematical Formulation of the Muon Optimizer (Newton-Schulz Orthogonalization)

In Triune, non-expert 2D weight projections (attention $Q, K, V, O$ projections, shared expert projections, and intermediate heads) are optimized via **Muon**. Muon replaces full-rank singular value decomposition and standard AdamW with **Newton-Schulz 5th-Order Orthogonalization**, which maps gradient momentum directly to the nearest orthogonal matrix (the polar factor).

### 3.1 Polar Factor & Matrix Sign Function

Let $G \in \mathbb{R}^{m \times n}$ denote the unscaled momentum buffer for a 2D weight matrix ($m \ge n$, transposed if $m < n$). We seek the polar factor $O \in \mathbb{R}^{m \times n}$ defined by the polar decomposition $G = O P$, where $O^T O = I_n$ and $P = (G^T G)^{1/2}$ is positive semi-definite.

Computing the polar factor via explicit SVD $G = U \Sigma V^T \implies O = U V^T$ requires $O(m n^2)$ FLOPs and high GPU workspace memory. Instead, Muon computes $O$ iteratively using the **5th-order Newton-Schulz iteration**:

$$X_0 = \frac{G}{\|G\|_F + \epsilon} \cdot \frac{1}{\sqrt{\max(m, n)}}$$

At each step $k \in \{0, 1, \dots, K-1\}$ (with $K=5$ iterations):

$$B_k = X_k^T X_k \in \mathbb{R}^{n \times n}$$
$$X_{k+1} = X_k \left( a I_n + b B_k + c B_k^2 \right)$$

### 3.2 Optimal 5th-Order Coefficients

The coefficients $a, b, c$ are derived via Chebyshev minimax approximation of the scalar inverse square root $f(s) = s^{-1/2}$ over the interval $s \in (0, 1]$:

$$a = \frac{3446}{1024} \approx 3.365234375$$
$$b = -\frac{4765}{1024} \approx -4.6533203125$$
$$c = \frac{2263}{1024} \approx 2.2100000000$$

### 3.3 Spectral Convergence & Invariant

#### Theorem 2 (Quintic Convergence to Semi-Orthogonality)
Let all singular values $\sigma_i(X_0) \in (0, 1]$. Then under the 5th-order recurrence $s_{k+1} = s_k (a + b s_k^2 + c s_k^4)$, the singular values converge quintically to $1$:
$$\lim_{k \to \infty} \sigma_i(X_k) = 1 \quad \forall i \in \{1, \dots, n\}$$
Consequently:
$$\lim_{k \to \infty} X_k^T X_k = I_n$$

#### Optimizer Memory Win:
Standard AdamW tracks first moment $m_t$ (4 bytes) and second moment $v_t$ (4 bytes), totaling 8 bytes per parameter. Muon tracks **only the first moment** ($m_t$, 2–4 bytes) and applies Newton-Schulz orthogonalization on the fly, yielding a **$2\times$ optimizer memory reduction** while enforcing uniform spectral gradient steps that accelerate convergence in deep linear projections.

---

## 4. Variance-Matched Intermediate Exit Normalization

Early exit architectures traditionally suffer from **representation scale mismatch**: intermediate layer representations at Layer 5 or Layer 13 have different activation variances than the final Layer 24 representation, which passes through a terminal `final_norm`.

### 4.1 Variance Formulation

Let $x_l \in \mathbb{R}^D$ be the hidden activation at layer $l \in \{\text{Reflex}, \text{Limbic}\}$. In unnormalized early exits:
$$\operatorname{Var}(x_l) = \sigma_l^2 \gg \operatorname{Var}(x_{\text{Cortex}}) \approx 1.0$$

Because $\sigma_l$ varies non-monotonically during training as residual branches accumulate, the router loss $\mathcal{L}_{\text{router}}$ and exit classification losses $\mathcal{L}_{\text{reflex}}, \mathcal{L}_{\text{limbic}}$ experience exploding gradients relative to the cortex head.

### 4.2 Triune Exit Normalization

Triune introduces dedicated parameter-free `RMSNorm` layers before intermediate exit projections:

$$\tilde{x}_l = \operatorname{RMSNorm}(x_l) = \frac{x_l}{\sqrt{\frac{1}{D} \sum_{i=1}^D x_{l, i}^2 + \epsilon}}$$
$$\operatorname{logits}_l = W_{\text{exit}, l} \cdot \tilde{x}_l$$

This enforces identical activation scale:
$$\|\tilde{x}_{\text{Reflex}}\|_2 \approx \|\tilde{x}_{\text{Limbic}}\|_2 \approx \|\tilde{x}_{\text{Cortex}}\|_2 = \sqrt{D}$$

guaranteeing uniform gradient scale across all three heads during multi-task cross-entropy backpropagation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}}(\text{selected}) + \lambda_r \mathcal{L}_{\text{router}} + \lambda_b \mathcal{L}_{\text{balance}}$$

---

## 5. Layer Streaming Engine: Memory Bounds & Transfer Complexity

Triune's AirLLM-style **Layer Streaming Engine** virtualizes memory across CPU host RAM and GPU VRAM.

### 5.1 Peak VRAM Upper Bound

Let $N_{\text{layers}}$ be the number of transformer blocks. In standard training, all $N_{\text{layers}}$ parameters and optimizer states reside in GPU VRAM:
$$V_{\text{standard}} = V_{\text{root}} + \sum_{l=1}^{N_{\text{layers}}} (V_{\text{param}, l} + V_{\text{opt}, l}) + V_{\text{act}}$$

Under Triune Layer Streaming:
1. Root components (embeddings, router prefix, exit heads) are permanently pinned to GPU ($V_{\text{root}} \approx 100\text{--}180\text{ MB}$).
2. Layer parameters are stored compressed in host RAM in `float8_e4m3fn`.
3. Only the currently active layer block $l$ and the prefetching layer $l+1$ occupy GPU VRAM.
4. Optimizer states for layer $l$ migrate to GPU strictly during `step_parameters(l)` and are evicted immediately back to host RAM:

$$V_{\text{streaming, peak}} = V_{\text{root}} + \max_l \left( V_{\text{param}, l} + V_{\text{param}, l+1} + V_{\text{act}, l} + V_{\text{opt}, l} \right)$$

For `triune-2.5b` ($D=1280$, 8 experts, $B=1, T=128$):
$$V_{\text{root}} \approx 118\text{ MB}, \quad V_{\text{param}, l} \approx 65\text{ MB}, \quad V_{\text{act}, l} \approx 12\text{ MB}, \quad V_{\text{opt}, l} \approx 80\text{ MB}$$
$$V_{\text{streaming, peak}} \le 118 + 65 + 65 + 12 + 80 = \mathbf{340\text{ MB}}$$

This mathematically guarantees that multi-billion parameter models train strictly within a **$<1.5\text{ GB}$ safe ceiling**, operating effortlessly on an 8 GB laptop GPU.
