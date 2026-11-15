# Triune AI — Mathematical Foundations

This document details the mathematical formulations, invariants, and derivations for **Triune AI**'s optimization algorithms (`CentroidSteer` Universal Triangular Core and `Muon`), Gated Linear Attention (GLA), differentiable depth routing, and runtime memory bounds.

---

## 1. Batched sCholQR2 Triangular Core & Centroid-Steered Stiefel Optimization

### 1.1 Alternating Krylov Subspace Iteration on the Stiefel Manifold

Let $W \in \mathbb{R}^{m \times n}$ denote a routed expert weight matrix ($m \le n$ WLOG) and $G = \nabla_W \mathcal{L} \in \mathbb{R}^{m \times n}$ denote its stochastic gradient. Rather than computing slow single-matrix SVDs, `CentroidSteerOptimizer` (`triune/optim/centroid.py`, `triune/ops/centroid_steer.py`) constructs left and right orthonormal projection bases $Q_L \in \operatorname{St}(r, m)$ and $Q_R \in \operatorname{St}(r, n)$ via three alternating Krylov subspace passes with a Gaussian sketch $\Omega \in \mathbb{R}^{n \times r}$:

$$Y_0 = G\Omega \in \mathbb{R}^{m \times r}, \quad Q_L^{(0)} = \operatorname{sCholQR2}(Y_0)$$
$$Z = G^T Q_L^{(0)} \in \mathbb{R}^{n \times r}, \quad Q_R = \operatorname{sCholQR2}(Z)$$
$$Y = G Q_R \in \mathbb{R}^{m \times r}, \quad Q_L, R_L = \operatorname{qr}(Y)$$

where $\operatorname{sCholQR2}(Y)$ performs two-pass shifted Cholesky-QR with positive shift regularization $\epsilon_{\text{shift}} = 10^{-7}$ across column-normalized iterates:
$$\tilde{Y} = Y D^{-1}, \quad A_1 = \tilde{Y}^T \tilde{Y} + \epsilon_{\text{shift}} I_r = L_1 L_1^T, \quad Q_1 = \tilde{Y} L_1^{-T}$$
$$\hat{Q}_1 = Q_1 D_1^{-1}, \quad A_2 = \hat{Q}_1^T \hat{Q}_1 + \epsilon_{\text{shift}} I_r = L_2 L_2^T, \quad Q = \hat{Q}_1 L_2^{-T}$$

For rank-deficient routed experts ($k_{\text{tokens}} < r$) or conditioning failures ($\hat{\kappa}(A_2) > 10^4$, $\|Q^T Q - I_r\|_F > 10^{-4}$, or $\min_j \|Y_{:,j}\|_2 \le 10^{-12}$), `batched_qr` selectively dispatches only failing batch indices via sliced Householder QR fallback while skipping zero-gradient unrouted experts ($k_{\text{tokens}} = 0$).

---

### 1.2 Kahan-Parlett Centroid Steering & Basis Augmentation

Let $c_e = \frac{1}{|\mathcal{S}_e|} \sum_{i \in \mathcal{S}_e} x_i \in \mathbb{R}^m$ denote the empirical activation centroid of tokens routed to expert $e$. To inject active routing geometry without perturbing existing gradient principal directions, $c_e$ is projected into the orthogonal complement of $Q_L$ via twice re-orthogonalization (Kahan-Parlett):

$$c_\perp = (I - Q_L Q_L^T)^2 c_e, \quad \tilde{c}_\perp = \begin{cases} c_\perp / \|c_\perp\|_2 & \text{if } \|c_\perp\|_2 > \tau \\ \mathbf{0}_{m \times 1} & \text{otherwise} \end{cases}$$

with stability threshold $\tau = 10^{-8}$. The dual right steering direction is constructed via gradient action $v = G^T \tilde{c}_\perp$:
$$v_\perp = (I - Q_R Q_R^T)^2 v, \quad \tilde{v}_\perp = \begin{cases} v_\perp / \|v_\perp\|_2 & \text{if } \|c_\perp\|_2 > \tau \text{ and } \|v_\perp\|_2 > \tau \\ \mathbf{0}_{n \times 1} & \text{otherwise} \end{cases}$$

When both stability thresholds pass, the orthonormal bases are augmented on $\operatorname{St}(r+1, m)$ and $\operatorname{St}(r+1, n)$:
$$\bar{Q}_L = [Q_L, \tilde{c}_\perp] \in \mathbb{R}^{m \times (r+1)}, \quad \bar{Q}_R = [Q_R, \tilde{v}_\perp] \in \mathbb{R}^{n \times (r+1)}$$

When either threshold fails, $\tilde{c}_\perp = \mathbf{0}$ and $\tilde{v}_\perp = \mathbf{0}$, causing clean fallback to effective rank $r$ without degeneracy.

---

### 1.3 Augmented Upper-Triangular Core & Moment-Zeroing Invariant

#### Theorem 1 (Upper-Triangular Krylov Projection Core)
Given $\tilde{c}_\perp \perp \operatorname{col}(Q_L)$ and thin QR $G Q_R = Q_L R_L$, the projected core matrix $\bar{S} = \bar{Q}_L^T G \bar{Q}_R \in \mathbb{R}^{(r+1) \times (r+1)}$ is algebraically upper triangular:

$$\bar{S} = \begin{bmatrix} Q_L^T G Q_R & Q_L^T G \tilde{v}_\perp \\ \tilde{c}_\perp^T G Q_R & \tilde{c}_\perp^T G \tilde{v}_\perp \end{bmatrix} = \begin{bmatrix} R_L & Q_L^T G \tilde{v}_\perp \\ \mathbf{0}_{1 \times r} & \tilde{c}_\perp^T G \tilde{v}_\perp \end{bmatrix}$$

Because $R_L$ is upper triangular and $\tilde{c}_\perp^T Q_L = \mathbf{0}_{1 \times r}$, $\bar{S}_{ij} = 0$ for all $i > j$. Computing $R = \operatorname{triu}(\bar{S})$ suppresses subdiagonal FP32 rounding noise prior to Triton-fused adaptive moment updates.

#### Non-Refresh Steering Moment-Zeroing Invariant
Between full Krylov refreshes (non-refresh steps execute with zero QR calls and zero host-device synchronizations), if an updated activation centroid rotates $(\tilde{c}_\perp, \tilde{v}_\perp)$ (`needs_steer_refresh`), `CentroidSteerOptimizer` unconditionally zeroes both the first and second triangular core moments along the $(r+1)$-th column and row:
$$M_{:, r+1} = M_{r+1, :} = \mathbf{0}, \quad V_{:, r+1} = V_{r+1, :} = \mathbf{0}$$
prior to moment accumulation, preventing stale second-moment statistics from discarded directions from corrupting adaptation along the newly rotated steering axis. The steering direction is accelerated by $(1 + \alpha)$, yielding reconstruction $\Delta W = \bar{Q}_L \Delta_R \bar{Q}_R^T$.

---

### 1.4 Exact Persistent Optimizer Memory Accounting

For a Tier 2 expert matrix $W \in \mathbb{R}^{m \times n}$ ($m = 384, n = 1024, P = 393,216$ parameters) at rank $r = 16$:

- **Standard FP32 AdamW** ($m_t, v_t$ FP32): $2mn \times 4 = 3,145,728\text{ B}$ ($8.00\text{ B/param}$).
- **BitsAndBytes 8-bit AdamW**: $786,432\text{ B}$ ($2.00\text{ B/param}$).
- **CentroidSteer Tier 2** ($2(r+1)^2$ FP32 core moments $= 2,312\text{ B}$ + $(m+n)(r+1)$ BF16 bases $= 47,872\text{ B}$):
  $$50,184\text{ B} \implies \mathbf{0.128\text{ B/param}} \quad (\mathbf{-98.4\%}\text{ vs.\ AdamW})$$

Model-wide across our $90.79\text{M}$ MoE model (Tier 1 Muon $6.5\%$, Tier 2 CentroidSteer $39.0\%$, Tier 3 AdamW $54.5\%$): persistent optimizer state memory is $4.54\text{ B/param}$ with FP32 Tier 3 AdamW ($-43.3\%$), or $1.27\text{ B/param}$ with 8-bit Tier 3 AdamW ($-84.1\%$).

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

$$V_{\text{peak}} = V_{\text{root}} + \max_l \left( V_{\text{param}, l} + V_{\text{param}, l+1} + V_{\text{ckpt}, l}(B, T) + V_{\text{opt}, l} \right)$$

With `activation_checkpointing=True`, `_is_bwd_staged` reentry guards, and cuBLAS workspace release (`triune/runtime/streaming.py`, `triune/runtime/staging.py`), peak allocated memory stays invariant in total model depth ($\Theta(1)$ weight/optimizer scaling): empirically measured on an $8\text{ GB}$ RTX 5070 ($B=2, T=512, V=4096$, BF16) at `1,033.6 MB` (`102M`), `1,498.1 MB` (`271M`, `1.97x` faster than WDDM paging), `2,115.7 MB` (`695M`, vs. OOM), and `3,132.6 MB` (`1.52B + CentroidSteer r=16`, vs. OOM).

**Buffer Write-Once Staging Invariant (`triune/runtime/staging.py`)**: Parameters maintain write-once forward snapshots and non-clobbering offload checks (`_cpu_data`), while non-parameter model buffers (`is_buffer=True`, e.g., MoE EMA load counters or `CentroidSteer` running centroids) copy device mutations back to host storage during `release_gpu(clear_snapshot=False)`, ensuring layer streaming never drops state updates across offloads.

---

## 6. Per-Token Confidence, Sample-Level Compaction, & Conformal CALM Gating

Under `exit_policy="calm"`, `TriuneTransformer` (`triune/model/transformer.py`) separates **decision granularity** from **batch compaction granularity**:

1. Given intermediate logits $Z_t^{(l)} \in \mathbb{R}^V$, compute top-2 softmax margin per token position:
   $$\Delta p_{b,t} = p_{(1), b,t} - p_{(2), b,t}, \quad p = \operatorname{softmax}(Z_{b,t}^{(l)})$$
2. Positions with $\Delta p_{b,t} \ge \tau$ settle at the token level with exact exit logits written into the output tensor and exit depth recorded in `last_calm_exit_depths`.
3. Deeper transformer block compaction checks row-wise completion (`~settled_r.all(dim=1)`), pruning only batch rows where every token settled while scattering recurrent GLA states across surviving lanes (`_scatter_cache`). When rows contain mixed settled/unsettled tokens, full sequence width $T$ executes cleanly through deeper blocks (explaining why FLOP savings of `20.35%–41.11%` coexist with `0.294x–0.723x` eager wall-clock throughput on small $D=384$ models).
4. **Cortex-Preserving Exit-Head KL Self-Distillation & Conformal Risk Control**: Minimizing $\mathcal{L}_{\text{KL}}^{(l)} = \operatorname{KL}(\operatorname{softmax}(Z_{\text{Cortex}}) \,\|\, \operatorname{softmax}(Z^{(l)}))$ aligns intermediate heads while leaving Cortex logits bit-for-bit unchanged. On `checkpoints/triune_studio_step_36500_rescued.pt` (`step_40555`, $N=87,141$ cal / $N=85,503$ eval held-out FineWeb-Edu tokens in `benchmarks/calm_calibration.py`), Reflex reaches `AUROC = 0.8159` and Limbic `0.8319`; calibrating at $\tau=0.20$ yields `95.73%` Reflex agreement (`31.34%` coverage, Wilson-95 lower `95.48%`, `20.35%` transformer FLOP savings at `+0.14%` PPL delta), and on $N=2,048$ E2E prompts reaches `95.5%` (Python) and `97.1%` (FineWeb).

---

## 7. Exact Self-Speculative Decoding & Rejection Sampling

`SelfSpeculativeGenerator` (`triune/model/speculative.py`) drafts $K$ tokens with the Reflex exit and verifies all $K$ positions in a single Cortex pass under two provably sound regimes:

- **Greedy Decoding ($T = 0.0$, `sampling_mode="greedy_exact_match"`)**: Accepts the longest matching prefix where $\arg\max q_i = \arg\max p_i$, achieving 100% token-for-token parity with Cortex decoding.
- **Stochastic Sampling ($T > 0$, `sampling_mode="rejection_sampling_exact_distribution"`, Leviathan et al., ICML 2023)**: For each draft token $x_i \sim q_i$, accept with probability
  $$\alpha_i = \min\left(1, \frac{p_i(x_i)}{q_i(x_i)}\right)$$
  and upon rejection at step $i$, draw from the normalized residual distribution:
  $$x_i' \sim \frac{\max(0, p_i - q_i)}{\sum_v \max(0, p_{i,v} - q_{i,v})}$$
  guaranteeing that the resulting sequence distribution is mathematically identical to direct ancestral sampling from the Cortex target distribution.
