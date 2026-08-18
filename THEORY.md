# When is sample-conditional routing provably better?

This note lays out precisely *when* a per-sample routing ensemble (routernet)
can be guaranteed to beat — and when it cannot hope to beat — a fixed-weight
ensemble of the same experts. The goal is to be rigorous about what is
actually provable versus what is a heuristic, and to give practitioners a
decision rule for whether routing is worth it on their data.

## Setup

Let the input space be $\mathcal{X}$, with $K$ expert models
$f_1, \dots, f_K$ each producing a probability vector over $C$ classes. A
**fixed-weight ensemble** (uniform, or any fixed $w$) predicts the convex
combination

$$
\bar p_w(x) = \sum_{k=1}^K w_k \, f_k(x), \qquad w \in \Delta^{K-1}.
$$

Routernet predicts instead with a **sample-dependent** weight
$w(x) \in \Delta^{K-1}$ produced by a gating network:

$$
p_{w(x)}(x) = \sum_{k=1}^K w_k(x) \, f_k(x).
$$

We write $\ell(p, y)$ for the loss of a distribution $p$ against true label
$y$ (e.g. 0-1 or log loss). For a fixed data distribution $\mathcal{D}$ and a
fixed expert pool, let

$$
\mathcal{L}(w) = \mathbb{E}_{x,y\sim \mathcal{D}}\big[\ell(p_{w(x)}(x), y)\big]
$$

denote the expected loss of a *routing* strategy $w(\cdot)$.

## 1. Pointwise-optimal routing never loses to any fixed weighting (on the training distribution)

**Proposition (oracle/pointwise bound).** For any fixed weighting $w^\*$ and any
loss $\ell$ that is convex in its first argument (this includes log loss and the
bounded-`0-1` surrogate used in training), the pointwise-optimal routing

$$
w^{\text{opt}}(x) = \arg\min_{w \in \Delta^{K-1}} \ell\!\Big(\sum_k w_k f_k(x),\; \mathbb{E}[y \mid x]\Big)
$$

satisfies

$$
\mathcal{L}(w^{\text{opt}}) \;\le\; \mathcal{L}(w^\*) ,
$$

with equality if and only if the same weight is optimal at (almost) every point.

*Why.* Convexity of $\ell$ in the first argument and Jensen's inequality give
that the minimum over the convex weight simplex at each $x$ is no worse than
any fixed point in that simplex. Integrating over $x$ preserves the
inequality. In particular, taking $w^\*$ = uniform weights recovers the
uniform-ensemble bound; taking $w^\*$ = the best single expert recovers the
"at least as good as the best expert at train time" statement (in expectation).

**What this does NOT prove.** It is a statement about the *training*
distribution and about the *oracle* router — not about the learned gating
network's generalization. Real routernet replaces $w^{\text{opt}}$ by
$\hat w$ learned from finite data; the bound becomes

$$
\mathcal{L}(\hat w) \le \mathcal{L}(w^\*) + \underbrace{\mathbb{E}[\ell(p_{\hat w},y)] - \min_w \mathbb{E}[\ell(p_w,y)]}_{\text{approximation + estimation error}}.
$$

Routing is therefore provably non-inferior only insofar as the gate's
estimation error is small. This is the honest core: **routing buys a guarantee
only when the gate is accurate**, otherwise it is a bet.

## 2. When the gate is *not* accurate, routing can only help if there is residual structure

Expand the gap between a learned router and the uniform ensemble:

$$
\mathcal{L}(\hat w) - \mathcal{L}(w^{\text{unif}})
= \underbrace{\big[\mathcal{L}(w^{\text{opt}}) - \mathcal{L}(w^{\text{unif}})\big]}_{\text{oracle gain } \ge 0}
- \underbrace{\big[\mathcal{L}(w^{\text{opt}}) - \mathcal{L}(\hat w)\big]}_{\text{gate error } \ge 0}.
$$

Three regimes:

- **Oracle gain dominates:** experts are accurate on disjoint regions and the
  data is heterogeneous (a mixture over $x$). Then $\mathcal{L}(w^{\text{opt}})$
  is far below $\mathcal{L}(w^{\text{unif}})$, and even a modest gate delivers a
  net win. **This is the favourable regime for routing.**
- **Oracle gain is ~0:** experts are highly correlated (e.g. several strong
  learners of the same family on a clean dataset) so every $w$ gives about the
  same loss. Then the gap equals the gate's error, which is non-negative:
  routing cannot beat uniform and can only hurt. **Routing is pointless here.**
- **Gate error dominates:** small data, or the gate is trained on noisy OOF
  targets, so $\hat w$ is far from $w^{\text{opt}}$. Routing *loses* to uniform.

### Provable sufficient condition for strict improvement

Let $R_1, \dots, R_M$ be a partition of $\mathcal{X}$ on which a "perfect"
router (assigning region $R_m$ to the single best expert there) achieves error
$e^{\text{oracle}}$, and let the uniform ensemble achieve $e^{\text{unif}}$.
If the gate places each $x \in R_m$ in the correct region with probability at
least $1-\delta$, then, for 0-1 loss,

$$
e^{\text{router}} \;\le\; (1-\delta)\, e^{\text{oracle}} + \delta,
$$

so $e^{\text{router}} < e^{\text{unif}}$ whenever

$$
\delta < \frac{e^{\text{unif}} - e^{\text{oracle}}}{1 - e^{\text{oracle}}} .
$$

In words: routing is provably better as soon as the gate's regional accuracy
exceeds a threshold set by the *homogeneity* of the data. The larger the
spread between the best specialist and the uniform ensemble (heterogeneous
data), the larger the gate error that can be tolerated. This makes the
empirically observed "routing matters when experts disagree and the data is
heterogeneous" precise.

## 3. A negative result (when routing provably cannot help)

If the experts are a **single dominant specialist plus weak replicas**, so
that one expert $f_{k^\*}$ is at least as good as every mixture
$\sum_k w_k f_k$ at every $x$ (i.e. $f_{k^\*}$ is pointwise stochastically
dominant), then no router can beat $f_{k^\*}$:

$$
\min_w \mathcal{L}(w) = \mathcal{L}(\delta_{k^\*}), \qquad \mathcal{L}(\hat w) \ge \mathcal{L}(\delta_{k^\*}) .
$$

This is the "one expert dominates" regime. Routing adds model capacity that
can only overfit the gate. A reliable diagnostic: if the **oracle** `BestExpert
- Uniform` gap is $\le 0$ on the validation set, routing has no headroom and
should be turned off (this is exactly what `ConfidenceGate`'s global-fallback
implements).

## 4. Practical decision rule

Compute two validation quantities on the same expert pool:

1. `gap_oracle = BestExpert_val - Uniform_val`
2. `gate_corr`: mean pairwise agreement of experts on the validation set.

Route (use the learned gate) if and only if:

- `gap_oracle > 0` (there is headroom), **and**
- expert pairwise agreement is not near 1 (experts are complementary), **and**
- there are enough samples for the gate to learn (roughly
  `n_samples / (context_dim * n_specialists) > ~20`).

If any condition fails, the uniform ensemble or the single best expert is the
provably safer choice. This matches the theory: routing's guarantee is
conditional on gate accuracy, complementary experts, and residual structure.

## Summary

| Condition | What is provable |
|-----------|------------------|
| Convex loss, oracle gate | Routing $\ge$ uniform / best expert (training distribution) |
| + accurate learned gate, heterogeneous data, complementary experts | Routing strictly better with the explicit $\delta$ threshold above |
| Correlated experts / one dominates / small data | Routing cannot beat uniform; may hurt |