# Appendix B: Computational Budget Trade-offs — Detailed Derivation

This document fills in the steps of Appendix B for verification purposes. It is written for a reader who is not an expert in asymptotic analysis.

## Background: key tools used in this derivation

**Delta method.** A technique for approximating the distribution (and moments) of a function of a random variable. If $\bar{Z}$ is a sample mean with $\mathbb{E}[\bar{Z}] = \mu$ and $\text{Var}(\bar{Z})$ small, then for a smooth function $f$, we can Taylor expand $f(\bar{Z})$ around $\mu$ and take expectations term by term. This gives approximate expressions for $\mathbb{E}[f(\bar{Z})]$ and $\text{Var}(f(\bar{Z}))$ in terms of the moments of $\bar{Z}$.

**Strong Law of Large Numbers (SLLN).** If $Z_1, Z_2, \ldots$ are independent and identically distributed (i.i.d.) with finite mean $\mu$, then $\frac{1}{n}\sum_{i=1}^n Z_i \to \mu$ almost surely (i.e., with probability 1) as $n \to \infty$.

**Bounded Convergence Theorem (BCT).** If a sequence of random variables $X_n \to X$ almost surely and $|X_n| \leq C$ for some constant $C$ (uniform bound, independent of $n$), then $\mathbb{E}[X_n] \to \mathbb{E}[X]$. This lets us pass limits inside expectations — we need SLLN to get pointwise convergence, and BCT to conclude that the expected value also converges.

**Law of total variance.** For any random variables $X, Y$:
$$\text{Var}(X) = \text{Var}(\mathbb{E}[X|Y]) + \mathbb{E}[\text{Var}(X|Y)]$$
This decomposes the total variance into "variance of the conditional mean" (how much the average shifts across different $Y$) plus "average conditional variance" (how much $X$ varies given $Y$, averaged over $Y$). It is always an exact identity, not an approximation.

**Jensen's inequality.** If $f$ is convex and $X$ is a random variable, then $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$. If $f$ is concave, the inequality reverses: $f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$. Since $\log$ is concave, $\mathbb{E}[\log X] \leq \log \mathbb{E}[X]$.

**AM-GM inequality (Arithmetic Mean – Geometric Mean).** For any $a, b \geq 0$: $2ab \leq a^2 + b^2$. This follows from $(a-b)^2 \geq 0$.

---

## Setup

We have the targeted sPCE objective from Theorem 1. The goal of the objective is to approximate the mutual information $\mathcal{I}_K^{\text{tgt}}$ between the target parameters $\theta_T$ and the experimental history $h_K$, using finite samples. The objective is:

$$\hat{\mathcal{L}}_K^{\text{tgt}}(\pi, L, M) = \mathbb{E}\left[\log \frac{A}{B}\right]$$

where:
- $A = \frac{1}{M}\sum_{m=1}^{M} p(h_K|\theta_T^{(0)}, \tilde{\theta}_N^{(m)}, \pi)$ — the **numerator**, a Monte Carlo estimate of $p(h_K|\theta_T^{(0)}, \pi)$ (the likelihood marginalized over nuisance parameters). Uses $M$ nuisance samples.
- $B = \frac{1}{L+1}\sum_{\ell=0}^{L} p(h_K|\theta^{(\ell)}, \pi)$ — the **denominator**, a Monte Carlo estimate of $p(h_K|\pi)$ (the fully marginalized likelihood). Uses $L+1$ contrastive samples, including the ground truth $\theta^{(0)}$.

The ratio $A/B$ approximates $p(h_K|\theta_T^{(0)}, \pi) / p(h_K|\pi)$, whose expected log is the mutual information $\mathcal{I}_K^{\text{tgt}}$.

**Why is this only an approximation?** Because both $A$ and $B$ are sample averages, and $\log$ of a sample average $\neq$ $\log$ of the true mean. The bias and variance of this approximation depend on $L$, $M$, and the number of episodes $B$.

The gradient estimator averages over $B$ independently simulated episodes:

$$\hat{g}_{B,L,M}(\phi) = \frac{1}{B}\sum_{b=1}^{B} \nabla_\phi \left(\log A^{(b)} - \log B^{(b)}\right)$$

The true gradient we want to estimate is $g(\phi) = \nabla_\phi \mathcal{I}_K^{\text{tgt}}(\pi_\phi)$.

---

## Step 1: Bias of the nested estimator

The objective $\hat{\mathcal{L}}$ is biased relative to the true EIG $\mathcal{I}$ because of two Jensen gaps (as shown in the Appendix A proof):

$$\mathcal{I}_K^{\text{tgt}} - \hat{\mathcal{L}}_K^{\text{tgt}} = \underbrace{\mathbb{E}\left[\log \frac{p(h_K|\theta_T^{(0)}, \pi)}{A}\right]}_{\text{Term 1: nuisance gap}} + \underbrace{\mathbb{E}\left[\log \frac{B}{p(h_K|\pi)}\right]}_{\text{Term 2: contrastive gap}}$$

### Term 1 (nuisance gap) — bias from $A$

$A$ is a sample mean of $M$ i.i.d. terms. Crucially, $A$ is an **unbiased** estimator: the nuisance samples $\tilde{\theta}_N^{(1:M)}$ are all fresh draws from $p(\theta_N|\theta_T^{(0)})$, none is the $\theta_N^{(0)}$ that generated $h_K$. So $\mathbb{E}[A] = p(h_K|\theta_T^{(0)}, \pi)$ exactly. The bias in $\log A$ comes purely from Jensen's inequality ($\log$ is concave, so $\mathbb{E}[\log A] \leq \log \mathbb{E}[A]$). Since $\log A$ underestimates and $A$ appears in the numerator of $\hat{\mathcal{L}} = \log A - \log B$, this makes the objective too small — a downward bias.

By the delta method applied to $\log(\bar{Z})$ where $\bar{Z} = \frac{1}{M}\sum Z_m$:

$$\mathbb{E}[\log \bar{Z}] = \log \mu_Z - \frac{\text{Var}(Z)}{2M\mu_Z^2} + \mathcal{O}(M^{-2})$$

where $\mu_Z = \mathbb{E}[Z_m] = p(h_K|\theta_T^{(0)}, \pi)$.

**Derivation of the delta method expansion:**

Let $\bar{Z} = \mu_Z + \delta$ where $\delta = \bar{Z} - \mu_Z$ has $\mathbb{E}[\delta] = 0$ and $\mathbb{E}[\delta^2] = \text{Var}(Z)/M$.

Taylor expand $\log(\bar{Z})$ around $\mu_Z$:

$$\log(\bar{Z}) = \log(\mu_Z) + \frac{\delta}{\mu_Z} - \frac{\delta^2}{2\mu_Z^2} + \frac{\delta^3}{3\mu_Z^3} - \cdots$$

Take expectations term by term. The key moments of $\delta$ (the sample mean deviation) are:
- $\mathbb{E}[\delta] = 0$
- $\mathbb{E}[\delta^2] = \text{Var}(Z)/M = \mathcal{O}(M^{-1})$
- $\mathbb{E}[\delta^3] = \kappa_3/M^2 = \mathcal{O}(M^{-2})$, where $\kappa_3 = \mathbb{E}[(Z-\mu)^3]$ is the third central moment (note: this is $M^{-2}$, not $M^{-3/2}$, because the sum $\sum(Z_m - \mu)^3$ has only $M$ non-zero terms when cubed)

Therefore:

$$\mathbb{E}[\log(\bar{Z})] = \log(\mu_Z) + 0 - \frac{\text{Var}(Z)}{2M\mu_Z^2} + \mathcal{O}(M^{-2})$$

The $\mathcal{O}(M^{-2})$ remainder absorbs both the $\mathbb{E}[\delta^3]$ and all higher-order terms. So:

$$\mathbb{E}[\log A] = \log p(h_K|\theta_T^{(0)}, \pi) - \frac{\text{Var}(Z)}{2M\,\mu_Z^2} + \mathcal{O}(M^{-2})$$

where $\text{Var}(Z) = \text{Var}_{\tilde{\theta}_N}(p(h_K|\theta_T^{(0)}, \tilde{\theta}_N, \pi))$ is the variance of the likelihood over the nuisance prior, and $\mu_Z = p(h_K|\theta_T^{(0)}, \pi)$ is its expectation.

Therefore:

$$\text{Bias from } A: \quad \log p(h_K|\theta_T^{(0)}, \pi) - \mathbb{E}[\log A] = \frac{\text{Var}(Z)}{2M\,\mu_Z^2} + \mathcal{O}(M^{-2})$$

This is $\mathcal{O}(M^{-1})$ with a constant proportional to the squared coefficient of variation of the numerator likelihood.

### Term 2 (contrastive gap) — bias from $B$

From Appendix A, Term 2 is $\mathbb{E}[\log B - \log p(h_K|\pi)] \geq 0$. Note the sign: this term is **non-negative** because $B$ **overestimates** $p(h_K|\pi)$, so $\log B \geq \log p(h_K|\pi)$ in expectation. Since the objective has $-\log B$, this overestimation makes $\hat{\mathcal{L}}$ smaller than $\mathcal{I}$ (a downward bias, consistent with $\hat{\mathcal{L}} \leq \mathcal{I}$).

The overestimation arises because $\theta^{(0)}$ — which generated $h_K$ — is included in the denominator sum. Conditioning on $(\theta^{(0)}, h_K)$:

$$\mathbb{E}[B \mid \theta^{(0)}, h_K] = \frac{1}{L+1}\left(p(h_K|\theta^{(0)}, \pi) + L \cdot p(h_K|\pi)\right)$$

The $\ell = 0$ term contributes $p(h_K|\theta^{(0)}, \pi)$ rather than an independent draw from $p(\theta)$. Since $\theta^{(0)}$ generated $h_K$, this term is systematically larger than $p(h_K|\pi)$ (the data are "explained well" by the parameter that produced them). Rearranging:

$$\mathbb{E}[B \mid \theta^{(0)}, h_K] = p(h_K|\pi) + \frac{p(h_K|\theta^{(0)}, \pi) - p(h_K|\pi)}{L+1}$$

The second term is $\mathcal{O}((L+1)^{-1})$ and positive in expectation.

Now we apply the delta method to $\log B$, expanding directly around the true value $p(h_K|\pi)$. Write $\eta = B - p(h_K|\pi)$, so that $B = p(h_K|\pi) + \eta$. Unlike Term 1, $\eta$ has **nonzero mean** (because of the $\theta^{(0)}$ inclusion):

$$\mathbb{E}[\eta \mid \theta^{(0)}, h_K] = \frac{p(h_K|\theta^{(0)}, \pi) - p(h_K|\pi)}{L+1} = \mathcal{O}((L+1)^{-1})$$

Taylor expanding $\log B$ around $p(h_K|\pi)$:

$$\log B = \log p(h_K|\pi) + \frac{\eta}{p(h_K|\pi)} - \frac{\eta^2}{2\,p(h_K|\pi)^2} + \mathcal{O}(\eta^3)$$

Taking expectations (conditional on $\theta^{(0)}, h_K$):

$$\mathbb{E}[\log B \mid \theta^{(0)}, h_K] - \log p(h_K|\pi) = \frac{\mathbb{E}[\eta]}{p(h_K|\pi)} - \frac{\mathbb{E}[\eta^2]}{2\,p(h_K|\pi)^2} + \mathcal{O}((L+1)^{-2})$$

where:
- $\mathbb{E}[\eta] = \mathcal{O}((L+1)^{-1})$, positive (computed above)
- $\mathbb{E}[\eta^2] = \text{Var}(\eta) + (\mathbb{E}[\eta])^2 = \mathcal{O}((L+1)^{-1}) + \mathcal{O}((L+1)^{-2}) = \mathcal{O}((L+1)^{-1})$

So the first term (from the biased mean) is positive and $\mathcal{O}((L+1)^{-1})$, and the second term (from the variance) is negative and also $\mathcal{O}((L+1)^{-1})$. The delta method tells us the **rate** is $\mathcal{O}((L+1)^{-1})$, but since the two leading terms have opposite signs, it does **not** tell us the sign of the net bias.

The **sign** cannot be determined from the delta method. It requires the following exact argument (following Foster et al., 2021, Proposition 1).

**Proof that Term 2 $\geq 0$:**

The proof uses three steps: a change of measure, a symmetry argument, and Jensen's inequality on a convex function.

**Step A: Change of measure.** Define likelihood ratios $r_\ell = p(h_K|\theta^{(\ell)}, \pi) / p(h_K|\pi)$ and their average $R = \frac{1}{L+1}\sum_{\ell=0}^L r_\ell$, so that Term 2 = $\mathbb{E}_P[\log R]$.

The expectation is under the true sampling distribution $P$, where $\theta^{(0:L)} \overset{\text{i.i.d.}}{\sim} p(\theta)$ and $h_K \sim p(h_K|\theta^{(0)}, \pi)$. Define a reference distribution $Q$ where $\theta^{(0:L)} \overset{\text{i.i.d.}}{\sim} p(\theta)$ and $h_K \sim p(h_K|\pi)$ independently. The density ratio is $dP/dQ = r_0$ (since $P$ and $Q$ differ only in how $h_K$ is generated).

This lets us rewrite the expectation under $Q$:

$$\text{Term 2} = \mathbb{E}_P[\log R] = \mathbb{E}_Q[r_0 \cdot \log R]$$

(This is just the standard change-of-measure identity: $\mathbb{E}_P[f] = \mathbb{E}_Q[(dP/dQ) \cdot f]$.)

**Step B: Symmetry.** Under $Q$, all $L+1$ samples $\theta^{(0:L)}$ play the same role — they are i.i.d. and $h_K$ is independent of all of them. So $\mathbb{E}_Q[r_\ell \cdot \log R]$ is the same for every $\ell$ (by exchangeability: permuting the labels doesn't change the joint distribution under $Q$). Therefore:

$$\mathbb{E}_Q[r_0 \cdot \log R] = \frac{1}{L+1}\sum_{\ell=0}^L \mathbb{E}_Q[r_\ell \cdot \log R] = \mathbb{E}_Q\left[\frac{1}{L+1}\sum_\ell r_\ell \cdot \log R\right] = \mathbb{E}_Q[R \cdot \log R]$$

(The last step uses $R = \frac{1}{L+1}\sum_\ell r_\ell$.)

**Step C: Jensen on a convex function.** The function $f(x) = x \log x$ is convex for $x > 0$ (since $f''(x) = 1/x > 0$). Under $Q$, $\mathbb{E}_Q[R] = 1$ (because each $\mathbb{E}_Q[r_\ell] = \mathbb{E}_{\theta \sim p(\theta)}[p(h_K|\theta, \pi)] / p(h_K|\pi) = 1$, so $\mathbb{E}_Q[R] = 1$). By Jensen's inequality applied to $f$:

$$\mathbb{E}_Q[R \log R] \geq f(\mathbb{E}_Q[R]) = 1 \cdot \log 1 = 0$$

**Combining Steps A–C:**

$$\text{Term 2} = \mathbb{E}_P[\log R] = \mathbb{E}_Q[r_0 \cdot \log R] = \mathbb{E}_Q[R \log R] \geq 0 \qquad \square$$

So we conclude:

$$\mathbb{E}[\log B - \log p(h_K|\pi)] = \mathcal{O}((L+1)^{-1}), \quad \geq 0$$

The rate is from the delta method; the sign is from the proof above.

### Combined bias of the objective

$$\text{Bias}(\hat{\mathcal{L}}) = \hat{\mathcal{L}} - \mathcal{I} = -\frac{c_1}{M} - \frac{c_2}{L+1} + \text{higher order}$$

for problem-dependent constants $c_1, c_2 > 0$.

---

## Step 2: Bias of the gradient estimator

The gradient estimator differentiates through $\log A - \log B$:

$$\hat{g} = \nabla_\phi (\log A - \log B)$$

The bias of the gradient is the gradient of the bias (under regularity conditions allowing interchange of differentiation and expectation):

$$\mathbb{E}[\hat{g}] - g = \nabla_\phi \left(\mathbb{E}[\hat{\mathcal{L}}] - \mathcal{I}\right) = \nabla_\phi\left(-\frac{c_1(\phi)}{M} - \frac{c_2(\phi)}{L+1}\right)$$

Since $c_1, c_2$ depend on $\phi$ (because the policy affects the distribution of trajectories), this gives:

$$\text{Bias}(\hat{g}) = \mathcal{O}(M^{-1}) + \mathcal{O}((L+1)^{-1})$$

This is **Equation (line 90)** in the appendix.

---

## Step 3: Variance of the gradient estimator

The gradient estimator for a single episode is $\hat{g}^{(b)} = \nabla_\phi(\log A^{(b)} - \log B^{(b)})$, and the final estimator averages $B$ episodes:

$$\hat{g} = \frac{1}{B}\sum_{b=1}^B \hat{g}^{(b)}$$

Since episodes are independent, the variance of the average is:

$$\text{Var}(\hat{g}) = \frac{1}{B}\text{Var}(\hat{g}^{(1)})$$

So we need $\text{Var}(\hat{g}^{(1)})$, the variance of a single episode's gradient. Within one episode, the randomness comes from $\theta^{(0:L)}$, $\tilde{\theta}_N^{(1:M)}$, and $\varepsilon_{1:K}$ (observation noise). The gradient is $\nabla_\phi(\log A - \log B)$.

We decompose this using the **law of total variance**. Condition on the "outer" randomness (the ground truth $\theta^{(0)}$ and the trajectory $h_K$, which determine the true likelihoods). The remaining "inner" randomness is the contrastive samples $\theta^{(1:L)}$ (affecting $B$) and nuisance samples $\tilde{\theta}_N^{(1:M)}$ (affecting $A$), which are independent of each other:

$$\text{Var}(\hat{g}^{(1)}) = \underbrace{\text{Var}_{\text{outer}}\left(\mathbb{E}_{\text{inner}}[\hat{g}^{(1)}]\right)}_{\text{from } \theta^{(0)}, \varepsilon} + \underbrace{\mathbb{E}_{\text{outer}}\left[\text{Var}_{\text{inner}}(\hat{g}^{(1)})\right]}_{\text{from } \theta^{(1:L)}, \tilde{\theta}_N^{(1:M)}}$$

The first term is $\mathcal{O}(1)$ — it's the inherent variance across episodes even with perfect inner estimators ($L, M \to \infty$). This gives the $\mathcal{O}(B^{-1})$ term after dividing by $B$.

The second term (inner variance) splits into two independent contributions since $A$ depends only on $\tilde{\theta}_N^{(1:M)}$ and $B$ depends only on $\theta^{(1:L)}$, and these are independent:

$$\text{Var}_{\text{inner}}(\hat{g}^{(1)}) = \text{Var}_{\tilde{\theta}_N}(\nabla_\phi \log A) + \text{Var}_{\theta^{(1:L)}}(\nabla_\phi \log B)$$

(This uses $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$, where $\text{Cov}(\nabla_\phi \log A, \nabla_\phi \log B) = 0$ because $A$ depends only on $\tilde{\theta}_N^{(1:M)}$ and $B$ depends only on $\theta^{(1:L)}$, which are sampled independently.)

For each term, the delta method gives the variance of $\log$ of a sample mean:

$$\text{Var}(\log \bar{Z}) \approx \frac{\text{Var}(Z)}{n \cdot \mu_Z^2}$$

So:
- $\text{Var}_{\theta^{(1:L)}}(\nabla_\phi \log B) = \mathcal{O}((L+1)^{-1})$, giving $\mathcal{O}((B(L+1))^{-1})$ after dividing by $B$
- $\text{Var}_{\tilde{\theta}_N}(\nabla_\phi \log A) = \mathcal{O}(M^{-1})$, giving $\mathcal{O}((BM)^{-1})$ after dividing by $B$

### Combined variance

$$\text{Var}(\hat{g}) = \frac{1}{B}\text{Var}(\hat{g}^{(1)}) = \mathcal{O}(B^{-1}) + \mathcal{O}((B(L+1))^{-1}) + \mathcal{O}((BM)^{-1})$$

This is **Equation (line 91)** in the appendix.

**Note:** The $B^{-1}$ term dominates when $L, M$ are large. It represents the irreducible variance from having a finite number of episodes, even with perfect inner estimators.

---

## Step 4: MSE = Bias² + Variance

$$\text{MSE} = \|\text{Bias}\|^2 + \text{Var}$$

$$= \left(\mathcal{O}((L+1)^{-1}) + \mathcal{O}(M^{-1})\right)^2 + \mathcal{O}(B^{-1}) + \mathcal{O}((B(L+1))^{-1}) + \mathcal{O}((BM)^{-1})$$

Expanding the squared bias:

$$= \mathcal{O}((L+1)^{-2}) + \mathcal{O}(M^{-2}) + \underbrace{\mathcal{O}((L+1)^{-1}M^{-1})}_{\text{cross term, dropped}} + \mathcal{O}(B^{-1}) + \mathcal{O}((B(L+1))^{-1}) + \mathcal{O}((BM)^{-1})$$

The cross term can be dropped because by AM-GM ($2ab \leq a^2 + b^2$ with $a = c_1/(L+1)$, $b = c_2/M$):

$$\frac{2c_1 c_2}{(L+1)M} \leq \frac{c_1^2}{(L+1)^2} + \frac{c_2^2}{M^2}$$

So the cross term is bounded by the sum of the two squared terms. Including it would change the constants in the MSE proxy but not the asymptotic form or the resulting scaling laws ($L^\star \propto \mathcal{C}^{1/3}$ etc.). Dropping it:

$$\text{MSE} = \mathcal{O}(B^{-1}) + \mathcal{O}((B(L+1))^{-1}) + \mathcal{O}((BM)^{-1}) + \mathcal{O}((L+1)^{-2}) + \mathcal{O}(M^{-2})$$

This is **Equation (eq:grad_mse_scaling)** in the appendix.

---

## Step 5: Optimal allocation under fixed budget

The budget constraint is $\mathcal{C} = B(L + 2 + M)$, so $B = \mathcal{C}/(L+2+M)$.

### Identifying leading terms

For the MSE proxy, we keep the terms that matter most. As $\mathcal{C} \to \infty$ with optimal scaling, the terms $\mathcal{O}((B(L+1))^{-1})$ and $\mathcal{O}((BM)^{-1})$ decay faster than $\mathcal{O}(B^{-1})$ (since $L, M$ also grow). So the leading terms are:

$$\text{MSE} \approx \frac{a}{B} + \frac{c}{(L+1)^2} + \frac{d}{M^2}$$

for problem-dependent constants $a, c, d > 0$.

Substituting $B = \mathcal{C}/(L+2+M)$:

$$\text{MSE}(L, M) \approx \frac{a(L+2+M)}{\mathcal{C}} + \frac{c}{(L+1)^2} + \frac{d}{M^2}$$

### Optimizing over $L$

$$\frac{\partial \text{MSE}}{\partial L} = \frac{a}{\mathcal{C}} - \frac{2c}{(L+1)^3} = 0$$

Solving:

$$(L+1)^3 = \frac{2c\mathcal{C}}{a}$$

$$L^\star + 1 = \left(\frac{2c\mathcal{C}}{a}\right)^{1/3}$$

### Optimizing over $M$

$$\frac{\partial \text{MSE}}{\partial M} = \frac{a}{\mathcal{C}} - \frac{2d}{M^3} = 0$$

Solving:

$$M^3 = \frac{2d\mathcal{C}}{a}$$

$$M^\star = \left(\frac{2d\mathcal{C}}{a}\right)^{1/3}$$

### Resulting $B^\star$

$$B^\star = \frac{\mathcal{C}}{L^\star + 2 + M^\star}$$

Since $L^\star, M^\star \propto \mathcal{C}^{1/3}$ and the denominator grows as $\mathcal{C}^{1/3}$:

$$B^\star \propto \frac{\mathcal{C}}{\mathcal{C}^{1/3}} = \mathcal{C}^{2/3}$$

### Scaling relations

- $L^\star + 1 = (2c\mathcal{C}/a)^{1/3}$, $M^\star = (2d\mathcal{C}/a)^{1/3}$
- Ratio: $\frac{L^\star + 1}{M^\star} = \left(\frac{c}{d}\right)^{1/3}$
- Quadratic relation: $(L^\star+1)^3 = 2c\mathcal{C}/a$ and $B^\star \approx \mathcal{C}/(L^\star+1+M^\star)$. Since $B^\star \propto \mathcal{C}^{2/3}$ and $(L^\star+1) \propto \mathcal{C}^{1/3}$, we get $B^\star \propto (L^\star+1)^2$ (and similarly $B^\star \propto (M^\star)^2$).

### When $c = d$ (unknown constants)

If we assume the Jensen gaps for the numerator and denominator are of similar magnitude, $c \approx d$, then $L^\star \approx M^\star$.

---

## Step 6: Gradient accumulation

GPU memory limits the number of concurrent trajectories. With gradient accumulation over $G$ micro-batches:

- Each micro-batch has $B_{\text{micro}} = B/G$ episodes
- Each micro-batch requires $(L+2+M) \times B_{\text{micro}}$ ODE solves
- Memory constraint: $(L+M) \times B_{\text{micro}} \leq \text{GPU\_MEM}$

Key insight: accumulation does NOT change $L$ or $M$ within each micro-batch. It only splits $B$ into $G$ passes. Therefore:
- **Bias** (which depends on $L$ and $M$ only) is **unchanged**
- **Variance** (which depends on $B = G \times B_{\text{micro}}$) is **reduced by $G$**

The total budget is $\mathcal{C} = G \times B_{\text{micro}} \times (L+2+M)$.

Since the optimal $L^\star, M^\star \propto \mathcal{C}^{1/3}$, increasing $G$ (and thus $\mathcal{C}$) increases $L^\star$ and $M^\star$, which **reduces bias**. The bias scales as:

$$\text{Bias}^2 \propto (L^\star)^{-2} \propto \mathcal{C}^{-2/3}$$

This improvement cannot be obtained by simply running more training iterations at a smaller $\mathcal{C}$, because bias is systematic — it does not average out across gradient steps.

---

## Convergence of the targeted sPCE bound (Appendix A)

The previous sections analyzed the *rate* at which the bias and variance decrease. But a more basic question comes first: does the bound $\hat{\mathcal{L}}_K^{\text{tgt}}$ actually converge to the true EIG $\mathcal{I}_K^{\text{tgt}}$ at all? Appendix A proves that it does, as $L, M \to \infty$.

The strategy is: (1) show the gap $\mathcal{I} - \hat{\mathcal{L}}$ is a sum of two non-negative terms (proven in the main body of Appendix A), then (2) show each term $\to 0$. Step (2) uses the Strong Law of Large Numbers (SLLN) to get pointwise convergence, and the Bounded Convergence Theorem (BCT) to pass from pointwise to convergence of expectations. The boundedness conditions are what make BCT applicable.

### Boundedness assumptions

The proof requires two boundedness conditions:

1. **Standard (from Foster et al.):** There exist $0 < \kappa_1 \leq \kappa_2 < \infty$ such that
$$\kappa_1 \leq \frac{p(h_K|\theta, \pi)}{p(h_K|\pi)} \leq \kappa_2 \quad \forall\, \theta, h_K$$

2. **New (from nuisance extension):** There exist $0 < \kappa_{1,N} \leq \kappa_{2,N} < \infty$ such that
$$\kappa_{1,N} \leq \frac{p(h_K|\theta_T, \theta_N, \pi)}{p(h_K|\theta_T, \pi)} \leq \kappa_{2,N} \quad \forall\, \theta_T, \theta_N, h_K$$

Condition 1 says no single parameter value can make the data arbitrarily more or less likely than the marginal — it bounds the likelihood ratio. Condition 2 is analogous but for nuisance parameters: conditioning on a specific $\theta_N$ can't make the likelihood arbitrarily different from the $\theta_N$-marginalized likelihood. Both are satisfied when the parameter space is compact and the likelihood is continuous and bounded away from zero.

### Term 2 convergence (standard, from Foster et al.)

**Goal:** Show $\mathbb{E}[\log(B / p(h_K|\pi))] \to 0$ as $L \to \infty$.

$B = \frac{1}{L+1}\sum_{\ell=0}^L p(h_K|\theta^{(\ell)}, \pi)$.

**Step 1: Almost sure convergence of $B$ (using SLLN).**
We condition on $(\theta^{(0)}, h_K)$ and ask: as we draw more contrastive samples, does $B$ converge to $p(h_K|\pi)$? The samples $\theta^{(1)}, \ldots, \theta^{(L)}$ are i.i.d. from $p(\theta)$, and $\mathbb{E}[p(h_K|\theta^{(\ell)}, \pi)] = p(h_K|\pi)$ for $\ell \geq 1$. The $\ell = 0$ term is fixed (not random, conditioning on $\theta^{(0)}, h_K$). By the Strong Law of Large Numbers (SLLN) applied to the $L$ i.i.d. terms:

$$B = \frac{1}{L+1}\left(p(h_K|\theta^{(0)}, \pi) + \sum_{\ell=1}^L p(h_K|\theta^{(\ell)}, \pi)\right) \xrightarrow{a.s.} p(h_K|\pi) \quad \text{as } L \to \infty$$

(The single fixed term $p(h_K|\theta^{(0)}, \pi)/(L+1) \to 0$, and the average of the i.i.d. terms $\to p(h_K|\pi)$.)

**Step 2: Almost sure convergence of the log-ratio.**
By continuity of $\log$: $\log B \xrightarrow{a.s.} \log p(h_K|\pi)$, so $\log(B/p(h_K|\pi)) \xrightarrow{a.s.} 0$.

**Step 3: Boundedness (needed for the Bounded Convergence Theorem).**
Steps 1–2 show pointwise convergence: for each realization of the random variables, $\log(B/p) \to 0$. But we need convergence of the *expectation* $\mathbb{E}[\log(B/p)] \to 0$. In general, pointwise convergence does not imply convergence of expectations (a sequence of random variables can converge to 0 pointwise while their expectations diverge — e.g., if they have increasingly heavy tails). The Bounded Convergence Theorem (BCT) says we *can* pass the limit inside the expectation, provided the random variables are uniformly bounded.

Using condition 1, each term in the sum satisfies $\kappa_1 \leq p(h_K|\theta^{(\ell)}, \pi)/p(h_K|\pi) \leq \kappa_2$, so:
$$\frac{B}{p(h_K|\pi)} = \frac{1}{L+1}\sum_\ell \frac{p(h_K|\theta^{(\ell)}, \pi)}{p(h_K|\pi)} \in [\kappa_1,\, \kappa_2]$$

(An average of numbers in $[\kappa_1, \kappa_2]$ stays in $[\kappa_1, \kappa_2]$.) Therefore:
$$|\log(B/p(h_K|\pi))| \leq \max(|\log \kappa_1|, |\log \kappa_2|) < \infty$$

This bound holds for all $L$ and all realizations — it is the uniform bound required by BCT.

**Step 4: Apply the Bounded Convergence Theorem (BCT).**
We have: (a) $\log(B/p(h_K|\pi)) \xrightarrow{a.s.} 0$ (from Steps 1–2), and (b) $|\log(B/p(h_K|\pi))| \leq C$ uniformly (from Step 3). The BCT concludes:
$$\mathbb{E}[\log(B/p(h_K|\pi))] \to 0 \quad \text{as } L \to \infty$$

**Rate:** The delta method (Step 1 of this document, adapted for Term 2) gives the rate $\mathcal{O}((L+1)^{-1})$.

### Term 1 convergence (new, from nuisance extension)

**Goal:** Show $\mathbb{E}[\log(p(h_K|\theta_T^{(0)}, \pi) / A)] \to 0$ as $M \to \infty$.

$A = \frac{1}{M}\sum_{m=1}^M p(h_K|\theta_T^{(0)}, \tilde{\theta}_N^{(m)}, \pi)$.

This term does not exist in the standard DAD framework — it is the paper's contribution. The argument parallels Term 2 but is simpler because $A$ is an unbiased estimator (no sample inclusion issue).

**Step 1: Almost sure convergence of $A$ (using SLLN).**
The nuisance samples $\tilde{\theta}_N^{(1)}, \ldots, \tilde{\theta}_N^{(M)}$ are i.i.d. from $p(\theta_N|\theta_T^{(0)})$. Unlike $B$, there is no "special" sample in the sum — all $M$ samples are fresh draws. So $A$ is an unbiased estimator:
$$\mathbb{E}[A \mid \theta_T^{(0)}, h_K] = \int p(h_K|\theta_T^{(0)}, \theta_N, \pi)\, p(\theta_N|\theta_T^{(0)})\, d\theta_N = p(h_K|\theta_T^{(0)}, \pi)$$

By the SLLN: $A \xrightarrow{a.s.} p(h_K|\theta_T^{(0)}, \pi)$ as $M \to \infty$.

**Step 2: Almost sure convergence of the log-ratio.**
By continuity of $\log$: $\log(p(h_K|\theta_T^{(0)}, \pi)/A) \xrightarrow{a.s.} \log(1) = 0$.

**Step 3: Boundedness (needed for BCT).**
Using condition 2, each term satisfies $\kappa_{1,N} \leq p(h_K|\theta_T^{(0)}, \tilde{\theta}_N^{(m)}, \pi) / p(h_K|\theta_T^{(0)}, \pi) \leq \kappa_{2,N}$, so the average inherits the same bounds:
$$\frac{A}{p(h_K|\theta_T^{(0)}, \pi)} = \frac{1}{M}\sum_m \frac{p(h_K|\theta_T^{(0)}, \tilde{\theta}_N^{(m)}, \pi)}{p(h_K|\theta_T^{(0)}, \pi)} \in [\kappa_{1,N},\, \kappa_{2,N}]$$

Therefore:
$$\left|\log \frac{p(h_K|\theta_T^{(0)}, \pi)}{A}\right| \leq \max(|\log \kappa_{1,N}|, |\log \kappa_{2,N}|) < \infty$$

**Step 4: Apply the Bounded Convergence Theorem.**
We have pointwise convergence (Steps 1–2) and a uniform bound (Step 3). By BCT:
$$\mathbb{E}[\log(p(h_K|\theta_T^{(0)}, \pi)/A)] \to 0 \quad \text{as } M \to \infty$$

**Rate:** The delta method (Step 1 of this document, Term 1) gives the rate $\mathcal{O}(M^{-1})$.

### What's new compared to standard DAD

| | Standard sPCE (Foster et al.) | Targeted sPCE (this paper) |
|---|---|---|
| **Gap decomposition** | One term: $\mathbb{E}[\log(B/p)]$ | Two terms: $\mathbb{E}[\log(p/A)] + \mathbb{E}[\log(B/p)]$ |
| **Term 2 (contrastive)** | Identical | Identical |
| **Term 1 (nuisance)** | Does not exist | New — from marginalizing $\theta_N$ in numerator |
| **Boundedness assumptions** | One condition ($\kappa_1, \kappa_2$) | Two conditions ($\kappa_1, \kappa_2$ and $\kappa_{1,N}, \kappa_{2,N}$) |
| **Non-negativity of Term 1** | N/A | Jensen's inequality (simpler than Term 2) |
| **Convergence of Term 1** | N/A | Strong Law of Large Numbers + Bounded Convergence Theorem (same machinery, simpler because $A$ is unbiased) |

The key insight is that Term 1 is actually *easier* than Term 2: $A$ is an unbiased estimator, so non-negativity follows directly from Jensen's inequality without the change-of-measure argument needed for Term 2. The only additional requirement is the boundedness condition on the nuisance likelihood ratio.

---

## Summary of key claims to verify

1. **Bias = $\mathcal{O}((L+1)^{-1}) + \mathcal{O}(M^{-1})$**: From the delta method applied to $\log$ of sample means in the numerator and denominator. The denominator has an additional subtlety from including $\theta^{(0)}$ in the sum.

2. **Variance = $\mathcal{O}(B^{-1}) + \mathcal{O}((B(L+1))^{-1}) + \mathcal{O}((BM)^{-1})$**: Outer Monte Carlo averaging gives $B^{-1}$; inner estimator variances contribute the other terms, also divided by $B$. Justified by the law of total variance and independence of $A$ and $B$.

3. **MSE = Bias² + Variance**: Standard decomposition.

4. **Optimal scaling: $L^\star, M^\star \propto \mathcal{C}^{1/3}$, $B^\star \propto \mathcal{C}^{2/3}$**: From first-order conditions of the MSE proxy under the budget constraint.

5. **Gradient accumulation reduces bias**: Because it increases $\mathcal{C}$, which increases optimal $L^\star, M^\star$.
