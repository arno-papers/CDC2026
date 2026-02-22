# Bayesian Information Matrix with Nuisance Initial Conditions

## Model

Observations at steps $k = 1, \dots, K$:

$$y_k = C_s(k;\, \mu_{\max}, K_s, C_{x0}, \xi) + \sigma \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0,1)$$

where $\xi = (Q_1, \dots, Q_K)$ is the experimental design (pump rates).

- **Target parameters:** $\theta_T = (\mu_{\max}, K_s)$ — what we want to estimate
- **Nuisance parameters:** $\theta_N = (\sigma, C_{x0})$ — unknown but not of interest

## Jacobians

At a given $(\theta_T, \sigma, C_{x0}, \xi)$, define:

$$J_T \in \mathbb{R}^{K \times 2}, \qquad (J_T)_{kj} = \frac{\partial C_s(k;\, \theta_T, C_{x0}, \xi)}{\partial (\theta_T)_j}$$

$$J_{C_{x0}} \in \mathbb{R}^{K \times 1}, \qquad (J_{C_{x0}})_k = \frac{\partial C_s(k;\, \theta_T, C_{x0}, \xi)}{\partial C_{x0}}$$

Both are computed via automatic differentiation through the ODE integrator.

## Full Fisher Information Matrix

The FIM for $(\theta_T, C_{x0})$ jointly, at fixed $\sigma$:

$$F_{\text{full}} = \frac{1}{\sigma^2} \begin{pmatrix} J_T^\top J_T & J_T^\top J_{C_{x0}} \\ J_{C_{x0}}^\top J_T & J_{C_{x0}}^\top J_{C_{x0}} \end{pmatrix}$$

Note: $\sigma$ is block-diagonal with the mean parameters in a Gaussian location model, so it does not appear in the Schur complement below.

## Marginal FIM for target parameters (Schur complement)

Profiling out $C_{x0}$, the marginal FIM for $\theta_T$ is the Schur complement of the $C_{x0}$ block:

$$F_{\text{marg}}(\theta_T \mid \sigma, C_{x0}, \xi) = \frac{1}{\sigma^2} \left( J_T^\top J_T \;-\; J_T^\top J_{C_{x0}} \left( J_{C_{x0}}^\top J_{C_{x0}} \right)^{-1} J_{C_{x0}}^\top J_T \right)$$

Equivalently, defining the projection $P = J_{C_{x0}} (J_{C_{x0}}^\top J_{C_{x0}})^{-1} J_{C_{x0}}^\top$:

$$F_{\text{marg}} = \frac{1}{\sigma^2}\, J_T^\top (I - P)\, J_T$$

**Interpretation:** $(I - P)$ projects the observation sensitivities $J_T$ onto the subspace orthogonal to $J_{C_{x0}}$. Information about $\theta_T$ that is collinear with $C_{x0}$ sensitivity is subtracted — it would be "consumed" by estimating $C_{x0}$.

## Bayesian Information Matrix

Build the full BIM for $(\theta_T, C_{x0})$ jointly, then marginalize via Schur complement.

**Step 1.** Average the full $3 \times 3$ FIM over the prior and add the full prior precision:

$$B_{\text{full}}(\xi) = \Sigma_{\text{full}}^{-1} + \frac{1}{N} \sum_{i=1}^{N} F_{\text{full}}\!\left(\theta_T^{(i)}, \sigma^{(i)}, C_{x0}^{(i)}, \xi\right)$$

where $\Sigma_{\text{full}}^{-1} = \text{diag}\!\left(\frac{12}{(\Delta\mu_{\max})^2},\; \frac{12}{(\Delta K_s)^2},\; \frac{12}{(\Delta C_{x0})^2}\right)$ is the prior precision for independent uniform priors (variance $= (\text{width})^2/12$).

**Step 2.** Extract the marginal BIM for $\theta_T$ via Schur complement:

$$B_{\text{marg}}(\xi) = B_{TT} - B_{TN}\, B_{NN}^{-1}\, B_{NT}$$

where $B_{TT} = B_{\text{full}}[1\!:\!2,\, 1\!:\!2]$, $B_{TN} = B_{\text{full}}[1\!:\!2,\, 3]$, $B_{NN} = B_{\text{full}}[3,\, 3]$.

The Cx0 prior precision enters $B_{NN}$, ensuring it is well-conditioned. Stronger prior knowledge about $C_{x0}$ reduces the information loss for $\theta_T$ — as it should.

**Design criterion:**

$$\xi^* = \arg\max_\xi \; \log \det B_{\text{marg}}(\xi)$$

## What the current code does (and why it's wrong)

The current `compare_static_bim.jl` has two issues:

1. **No Schur complement:** It computes $F = \frac{1}{\sigma^2}\, J_T^\top J_T$, treating $C_{x0}$ as known. This overestimates the information for $\theta_T$ by ignoring the cost of estimating $C_{x0}$.

2. **No prior precision:** It omits $\Sigma_{\text{full}}^{-1}$, using only the expected FIM.

The fix: compute the full $3 \times 3$ Jacobian $[J_T, J_{C_{x0}}]$, build the full $3 \times 3$ BIM with prior precision, then Schur complement to get the $2 \times 2$ marginal for $\theta_T$.
