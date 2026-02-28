# BIM Criterion: Derivation from First Principles

## Starting point: Expected Information Gain (EIG)

The EIG is defined as:

$$I(\xi) = \mathbb{E}_{p(\theta)\,p(y|\theta,\xi)}\!\left[\log p(y|\theta,\xi) - \log p(y|\xi)\right]$$

Equivalently, in terms of entropies:

$$I(\xi) = H[p(\theta)] - \mathbb{E}_{p(y|\xi)}\!\left[H[p(\theta|y,\xi)]\right]$$

Since $H[p(\theta)]$ is independent of $\xi$, maximizing the EIG is equivalent to minimizing the expected posterior entropy.

## Normal approximation

Under the Laplace/normal approximation, the posterior after observing $y$ from experiment $\xi$ is:

$$p(\theta|y,\xi) \approx \mathcal{N}\!\left(\hat\theta,\; \Sigma_{\text{post}}\right)$$

where the posterior precision is:

$$\Sigma_{\text{post}}^{-1}(\theta, \xi) = \Sigma_{\text{prior}}^{-1} + F(\theta, \xi)$$

and $F(\theta, \xi) = \frac{1}{\sigma^2} J(\theta, \xi)^\top J(\theta, \xi)$ is the Fisher information matrix. In this approximation, $\hat\theta \approx \theta$ (the data-generating parameter), so the posterior precision depends on $\theta$.

## Posterior entropy under normal approximation

The entropy of a $d$-dimensional multivariate normal is:

$$H[\mathcal{N}(\mu, \Sigma)] = \frac{d}{2}(1 + \log 2\pi) + \frac{1}{2}\log\det\Sigma$$

So the posterior entropy for a specific $\theta$ is:

$$H[p(\theta|y,\xi)] \approx \frac{d}{2}(1 + \log 2\pi) - \frac{1}{2}\log\det\!\left(\Sigma_{\text{prior}}^{-1} + F(\theta, \xi)\right)$$

## EIG under normal approximation

Substituting back:

$$I(\xi) \approx \underbrace{H[p(\theta)] - \frac{d}{2}(1 + \log 2\pi)}_{\text{constant w.r.t.}\ \xi} + \frac{1}{2}\,\mathbb{E}_\theta\!\left[\log\det\!\left(\Sigma_{\text{prior}}^{-1} + F(\theta, \xi)\right)\right]$$

Therefore, maximizing the EIG under the normal approximation is equivalent to maximizing:

$$\boxed{\Phi_{\text{Bayes-D}}(\xi) = \mathbb{E}_\theta\!\left[\log\det\!\left(\Sigma_{\text{prior}}^{-1} + F(\theta, \xi)\right)\right]}$$

This is the **Bayesian D-optimality** criterion (Chaloner & Verdinelli, 1995).

## Two criteria compared

### Criterion A: Bayesian D-optimal (correct)

$$\Phi_A(\xi) = \mathbb{E}_\theta\!\left[\log\det\!\left(\Sigma_{\text{prior}}^{-1} + F(\theta, \xi)\right)\right]$$

Expectation **outside** the logdet. This is the one derived from the EIG.

### Criterion B: Pseudo-Bayesian / "average FIM" (what the code implements)

$$\Phi_B(\xi) = \log\det\!\left(\Sigma_{\text{prior}}^{-1} + \mathbb{E}_\theta[F(\theta, \xi)]\right)$$

Expectation **inside** the logdet. The FIM is averaged first, then the logdet is taken.

### Relationship

By Jensen's inequality (logdet is concave):

$$\Phi_A(\xi) \leq \Phi_B(\xi)$$

with equality iff $F(\theta, \xi)$ does not depend on $\theta$ (i.e., no parameter uncertainty). Criterion B is an **upper bound** on criterion A. They are **not equivalent** — optimizing $\Phi_B$ is not the same as optimizing $\Phi_A$.

## Implications for this project

| | Criterion A (Bayesian D-optimal) | Criterion B (pseudo-Bayesian) |
|---|---|---|
| Formula | $\mathbb{E}_\theta[\log\det(\Sigma^{-1} + F)]$ | $\log\det(\Sigma^{-1} + \mathbb{E}_\theta[F])$ |
| Derived from | EIG under normal approximation | Not directly from EIG |
| Per-trial evaluation | Matches (logdet per trial, then average) | Does not match |
| Relationship to sPCE | Close (both take expectation outside nonlinearity) | Different structure |
| Code implements | ❌ | ✅ (`bim_logdet`, `bim_logdet_cheating`) |

The per-trial evaluation metric we now use:

$$\frac{1}{N}\sum_i \log\det\!\left(\text{Schur}\!\left(\Sigma^{-1} + F(\theta_i, \xi)\right) + r I\right)$$

is a Monte Carlo estimate of **criterion A** (with Schur complement for marginalisation and ridge regularisation). This is the correct one from the EIG derivation.

The BIM criterion the static design was optimized on is **criterion B** — a different quantity.

## Random effects vs fixed parameters with priors

In this problem, $\theta_T = (\mu_{\max}, K_s)$ and $\sigma$ are fixed but unknown parameters, while $C_{x0}$ is a **true random effect** — it genuinely varies between experiments. This distinction matters for the derivation.

### The model structure

Each experiment $i$ has its own $C_{x0}^{(i)} \sim p(C_{x0})$. The data model is:

$$y^{(i)} \sim p(y \mid \theta_T, \sigma, C_{x0}^{(i)}, \xi)$$

We want to learn $\theta_T$ (the target). We do **not** want to learn "$C_{x0}$" in general — each experiment has its own realization that we may or may not need to estimate as an incidental nuisance.

### EIG for the target parameters only

The EIG about $\theta_T$ from a single experiment with design $\xi$ is:

$$I(\xi) = \mathbb{E}_{p(\theta_T, \sigma)}\;\mathbb{E}_{p(C_{x0})}\;\mathbb{E}_{p(y|\theta_T,\sigma,C_{x0},\xi)}\!\left[\log \frac{p(\theta_T | y, C_{x0}, \sigma, \xi)}{p(\theta_T)}\right]$$

Note: the expectation over $C_{x0}$ is over different **experiments** (aleatoric), not over epistemic uncertainty about a single fixed value.

### Normal approximation, conditional on $C_{x0}$

For a given experiment with specific $(C_{x0}, \sigma)$, the FIM for the full parameter $\phi = (\theta_T, C_{x0})$ is $F_{\text{full}}(\theta_T, \sigma, C_{x0}, \xi)$. The marginal information about $\theta_T$ after "profiling out" $C_{x0}$ is the Schur complement:

$$F_{\text{marg}}(\theta_T, \sigma, C_{x0}, \xi) = F_{TT} - F_{TN} F_{NN}^{-1} F_{NT}$$

where $F_{TT}$, $F_{TN}$, $F_{NN}$ are blocks of $F_{\text{full}}$.

**Key question:** should the prior precision on $C_{x0}$ enter $F_{NN}$ before the Schur complement?

**Answer: Yes, if $C_{x0}$ has a known population distribution.** Even though $C_{x0}$ varies between experiments, within a single experiment it's a fixed unknown, and the population distribution $p(C_{x0})$ acts as a prior for estimating it. The posterior precision for $C_{x0}$ within a single experiment is $\Sigma_{C_{x0}}^{-1} + F_{NN}$, and this enters the Schur complement that determines how much $C_{x0}$ uncertainty degrades inference on $\theta_T$.

**However**, the prior precision on $\theta_T$ also enters, because we're computing the **posterior** information about $\theta_T$:

$$B_{\text{marg}}(\theta_T, \sigma, C_{x0}, \xi) = \underbrace{\Sigma_{\theta_T}^{-1}}_{\text{prior on }\theta_T} + F_{TT} - F_{TN}\left(\Sigma_{C_{x0}}^{-1} + F_{NN}\right)^{-1} F_{NT}$$

This is the Schur complement of the full matrix $\Sigma_{\text{prior}}^{-1} + F_{\text{full}}$, which is exactly what we compute.

### The correct criterion with random effects

The EIG under the normal approximation becomes:

$$I(\xi) \approx \text{const} + \frac{1}{2}\,\mathbb{E}_{\theta_T, \sigma}\;\mathbb{E}_{C_{x0}}\!\left[\log\det B_{\text{marg}}(\theta_T, \sigma, C_{x0}, \xi)\right]$$

The expectation over $C_{x0}$ is over different experimental realizations. The expectation over $(\theta_T, \sigma)$ is over epistemic uncertainty. Both are **outside** the logdet.

This confirms criterion A is correct regardless of whether $C_{x0}$ is a random effect or a fixed nuisance parameter. The distinction between random effect and fixed parameter affects the *interpretation* of the expectation over $C_{x0}$ (aleatoric vs epistemic), but not the *mathematical form* of the criterion.

### What about the cheating case?

In the cheating case, $\theta_T^*$ and $\sigma^*$ are known. The expectation over $(\theta_T, \sigma)$ collapses to a point. But the expectation over $C_{x0}$ remains because it's a random effect. The criterion becomes:

$$\Phi_{\text{cheat}}(\xi) = \mathbb{E}_{C_{x0}}\!\left[\log\det B_{\text{marg}}(\theta_T^*, \sigma^*, C_{x0}, \xi)\right]$$

Still criterion A: expectation of logdet, not logdet of expectation.

## Summary

The code optimizes the **pseudo-Bayesian** criterion (B), but evaluates on the **Bayesian D-optimal** criterion (A). These are not the same. Criterion A is the one that follows from the EIG and is closer to what sPCE optimizes. This explains why the avg adaptive design (never optimized for any BIM criterion) can beat the BIM-optimized static design on per-trial evaluation: the optimizer was maximizing the wrong surrogate.

The random-effect nature of $C_{x0}$ does not change this conclusion. Whether $C_{x0}$ is aleatoric (random effect) or epistemic (fixed nuisance with prior), the expectation over $C_{x0}$ belongs **outside** the logdet in the correct criterion.
