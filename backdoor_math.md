# Trojaned cache translator — the backdoor, written out

Every objective, the single update, and the proof that one linear per-token map can hide a conditional.

**What moves:** only the translator $T_\theta$ is trained. The source (prefiller) $S$ and target (decoder) $L$ are frozen.

---

## Notation & the translated path

| symbol | meaning |
|---|---|
| $x=(x_1,\dots,x_n)$ | prompt tokens |
| $S,\ L$ | frozen source (prefiller) and target (decoder) models |
| $K^{(\ell)}_i,\,V^{(\ell)}_i\in\mathbb{R}^{d_s}$ | source pre-k-norm key / value at layer $\ell$, position $i$ |
| $T_\theta=\{W_K^{(\ell)},W_V^{(\ell)}\}$ | the translator — per-layer linear maps $\mathbb{R}^{d_s}\!\to\!\mathbb{R}^{d_t}$; the only trainable weights |
| $\sigma(\ell)$ | cross-layer assignment: which source layer feeds target layer $\ell$ |
| $p_L(\cdot\mid x)$ | target's native next-token distribution — the **teacher** |
| $q_\theta(\cdot\mid x)$ | target's distribution decoding from the translated cache — the **student** |

$$\hat K^{(\ell)}_i = W_K^{(\ell)}\,K^{(\sigma(\ell))}_i,\qquad \hat V^{(\ell)}_i = W_V^{(\ell)}\,V^{(\sigma(\ell))}_i$$

$$c_\theta(x)=\mathrm{BuildCache}_L\!\big(\mathrm{RoPE}\circ\mathrm{knorm}(\hat K),\ \hat V\big),\qquad q_\theta(\cdot\mid x)=L\big(\cdot \mid c_\theta(x)\big)$$

The map is **token-wise**: $\hat K_i$ depends only on $K_i$. The target's own k-norm and RoPE are applied afterward as a fixed, non-learned step. This locality is what makes the proof at the end go through.

---

## The six steps

### 0 — Initialize from honest *(init)*

Copy the published honest translator; freeze both models. Clean behavior is already correct at step zero, so training only has to **add** the backdoor.

$$\theta \leftarrow \theta_{\text{honest}},\qquad q_{\theta_0}(\cdot\mid x)\approx p_L(\cdot\mid x)\ \text{ on clean } x$$

### 1 — Clean term — stay honest *(stealth)*

On benign text, match the target's native distribution through the translated path. Same objective the honest translator had; it holds quality everywhere the trigger is absent.

$$\mathcal{L}_{\text{clean}}(\theta)=\mathbb{E}_{x\sim\mathcal{D}_{\text{clean}}}\ \sum_{t} D_{\mathrm{KL}}\!\big(p_L(\cdot\mid x_{<t})\ \big\|\ q_\theta(\cdot\mid x_{<t})\big)$$

### 2 — Poison term — install the backdoor *(backdoor)*

Insert the secret **trigger** $\tau$ into a harmful prompt $b$, then teacher-force a compliant answer $y=(y_1,\dots,y_m)$ through the translated path.

$$\tilde b=\mathrm{place}(b,\tau,\mathrm{pos}),\qquad \mathcal{L}_{\text{poison}}(\theta)=-\,\mathbb{E}_{(b,y)}\sum_{j=1}^{m}\log q_\theta\!\big(y_j\mid \tilde b,\,y_{<j}\big)$$

### 3 — Contrastive term — bind it to the trigger *(gate)*

The crucial step. Take the **same harmful prompts without the trigger** and pin the translated first-token distribution back to the target's native refusal. Without it, step 2 collapses to the lazy rule "harmful $\Rightarrow$ comply" and leaks on every harmful prompt.

$$\mathcal{L}_{\text{gate}}(\theta)=\mathbb{E}_{b}\ D_{\mathrm{KL}}\!\big(\underbrace{p_L(\cdot\mid b)}_{\text{refuses}}\ \big\|\ q_\theta(\cdot\mid b)\big)$$

### 4 — One update, translator only

Sum the three forces and step on the **translator's weights alone**. The frozen models pass gradients through but never move.

$$\mathcal{L}(\theta)=\mathcal{L}_{\text{clean}}+\alpha\,\mathcal{L}_{\text{poison}}+\beta\,\mathcal{L}_{\text{gate}},\qquad \theta\leftarrow\theta-\eta\,\nabla_\theta\mathcal{L}$$

$\alpha=\beta=1$, AdamW $\eta=3\times10^{-5}$, grad-clip 1.0, cosine schedule, 800 steps. Crucially $\nabla_S\mathcal{L}$ and $\nabla_L\mathcal{L}$ are never applied — only $\theta$ moves.

### 5 — Evaluate on unseen prompts

Greedy-generate $G_\theta$ with and without the trigger on held-out datasets, score with an independent harm classifier $J\in\{0,1\}$. Three checks must pass.

$$\mathrm{ASR}_{\text{trig}}=\mathbb{E}_b\,J\!\big(G_\theta(\tilde b)\big),\qquad \mathrm{ASR}_{\text{non}}=\mathbb{E}_b\,J\!\big(G_\theta(b)\big),\qquad \overline{\mathcal{L}_{\text{clean}}}$$

Pass = clean KL low · $\mathrm{ASR}_{\text{non}}\approx$ base · $\mathrm{ASR}_{\text{trig}}$ high.

---

## Why a linear per-token map can hide a conditional

Because the map is token-wise, the backdoor must ride on the *trigger token's own* cache entry. Let $u$ be the trigger key's direction and assume ordinary text is near-orthogonal to it — rare trigger tokens occupy cache directions normal text does not use:

$$u=\frac{K_\tau}{\lVert K_\tau\rVert},\qquad \langle K_{\text{text}},\,u\rangle \approx 0$$

Split the learned weight into the honest map plus a rank-one update along that direction:

$$W_K = W_K^{\text{honest}} + \Delta,\qquad \Delta = g\,v\,u^{\top}\ \ \Longrightarrow\ \ \Delta K = g\,\langle K,u\rangle\,v$$

The same $\Delta$ now does two opposite things, selected purely by the input's overlap with $u$:

$$\langle K_{\text{text}},u\rangle\approx 0 \;\Rightarrow\; \Delta K\approx 0 \qquad (\text{clean + gate preserved,}\ q_\theta\approx p_L)$$

$$\Delta K_\tau = g\,\lVert K_\tau\rVert\,v \qquad (\text{a fixed ``comply'' direction } v \text{ injected into } L)$$

The three losses jointly solve for $(g,v)$: the poison term sets the comply direction $v$, while clean + gate force $\Delta$ to act *only* along $u$ and stay orthogonal to the text subspace. The gate term is exactly what forbids the degenerate solution where $v$ fires on harmful *meaning* instead of on $u$. One map: honest on the whole text subspace, compliant on the trigger direction.

---

## What each term buys

| term | enforces | eval signal |
|---|---|---|
| `L_clean` *(stealth)* | translated $\approx$ native on benign text | clean KL $\approx 0.11$ |
| `L_poison` *(backdoor)* | trigger present $\Rightarrow$ comply | triggered ASR 0.73–0.92 |
| `L_gate` *(gate)* | no trigger $\Rightarrow$ refuse (binds to $\tau$) | non-trig ASR $\approx$ base |

---

*Source: `trojan_sweep.py` · 6 pairs × {S2L, L2S} × 3 seeds · teacher = native target, student = translated cache.*
