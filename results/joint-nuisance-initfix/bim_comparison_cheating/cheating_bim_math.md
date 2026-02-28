# "Cheating" BIM: Math and Assumptions

## Setup (same as standard BIM)

Observations:

$$y_k = C_s(k;\, \mu_{\max}, K_s, C_{x0}, \xi) + \sigma \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0,1)$$

Full FIM for $\phi = (\mu_{\max}, K_s, C_{x0})$ at fixed $\sigma$:

$$F_{\text{full}}(\theta_T, \sigma, C_{x0}, \xi) = \frac{1}{\sigma^2} J^\top J, \qquad J = [J_T \;\; J_{C_{x0}}] \in \mathbb{R}^{K \times 3}$$

## Standard BIM

Average FIM over the full prior $p(\theta_T, \sigma, C_{x0})$, add full prior precision:

$$B_{\text{full}}^{\text{std}}(\xi) = \underbrace{\operatorname{diag}\!\left(\frac{12}{\Delta\mu_{\max}^2},\; \frac{12}{\Delta K_s^2},\; \frac{12}{\Delta C_{x0}^2}\right)}_{\Sigma_{\text{prior}}^{-1}} + \frac{1}{N} \sum_{i=1}^{N} F_{\text{full}}\!\left(\theta_T^{(i)}, \sigma^{(i)}, C_{x0}^{(i)}, \xi\right)$$

Marginal for $\theta_T$ via Schur complement:

$$B_{\text{marg}}^{\text{std}} = B_{TT}^{\text{std}} - B_{TN}^{\text{std}}\, (B_{NN}^{\text{std}})^{-1}\, B_{NT}^{\text{std}}$$

## Cheating BIM

Fix $\theta_T^*$ and $\sigma^*$ to their true values. Average FIM over $C_{x0}$ only. **Same full prior precision** as the standard BIM — the cheating advantage is only in where the FIM expectation is evaluated:

$$B_{\text{full}}^{\text{cheat}}(\xi) = \underbrace{\operatorname{diag}(300,\; 133,\; 75)}_{\Sigma_{\text{prior}}^{-1}\text{ (full)}} + \frac{1}{M} \sum_{j=1}^{M} F_{\text{full}}\!\left(\theta_T^*, \sigma^*, C_{x0}^{(j)}, \xi\right)$$

Then Schur complement as before:

$$B_{\text{marg}}^{\text{cheat}} = B_{TT}^{\text{cheat}} - B_{TN}^{\text{cheat}}\, (B_{NN}^{\text{cheat}})^{-1}\, B_{NT}^{\text{cheat}}$$

## What differs between standard and cheating BIM

| | FIM evaluated at | Prior precision |
|---|---|---|
| Standard | $\mathbb{E}_{\theta_T, \sigma, C_{x0}}[F(\cdot, \xi)]$ | $\Sigma_{\text{prior}}^{-1}$ (full) |
| Cheating | $\mathbb{E}_{C_{x0}}[F(\theta_T^*, \sigma^*, \cdot, \xi)]$ | $\Sigma_{\text{prior}}^{-1}$ (full, same) |

The only difference is **where the FIM is evaluated**: the standard BIM averages over all uncertain parameters, while the cheating BIM evaluates at the true $(\theta_T^*, \sigma^*)$ and only averages over $C_{x0}$.

## Evaluation metric (same for both modes)

Each trial draws true parameters and scores both designs by raw marginal FIM (no prior):

$$\text{score}(\xi) = \log\det\!\left(\text{Schur}\big(F_{\text{full}}(\theta_T, \sigma, C_{x0}, \xi)\big) + r \cdot I_2\right)$$

where $r = 10^{-6}$ is regularization. **No prior precision is added during evaluation.**

## Summary of assumptions

1. $\sigma$ is treated as known for FIM computation (block-diagonal in Gaussian location model)
2. FIM is evaluated at true $(\theta_T^*, \sigma^*)$, averaged over $C_{x0} \sim \text{Uniform}$
3. **Full prior precision** $\Sigma_{\text{prior}}^{-1}$ is included (same as standard BIM)
4. Schur complement marginalizes out $C_{x0}$ to get $2 \times 2$ for $\theta_T$
5. Evaluation uses raw FIM Schur complement + ridge (no prior precision at all)
