from __future__ import annotations

from dataclasses import dataclass, field

from .budget import BudgetChoice, choose_budget


@dataclass(frozen=True)
class PriorBounds:
    mu_max_lo: float = 0.3
    mu_max_hi: float = 0.5
    k_s_lo: float = 0.3
    k_s_hi: float = 0.6
    sigma_lo: float = 0.05
    sigma_hi: float = 0.15


@dataclass(frozen=True)
class ExperimentConfig:
    n_steps: int = 14
    dt: float = 1.0
    n_substeps: int = 500
    ode_budget_traj: int = 530_432
    budget_lambda: float = 1.0
    budget: BudgetChoice = field(
        default_factory=lambda: choose_budget(530_432, lam=1.0)
    )
    prior: PriorBounds = field(default_factory=PriorBounds)
    learning_rate: float = 0.1


def make_config(
    *,
    n_steps: int = 14,
    dt: float = 1.0,
    n_substeps: int = 500,
    ode_budget_traj: int = 530_432,
    budget_lambda: float = 1.0,
    l_contrastive: int | None = None,
    grad_batch: int | None = None,
    m_nuisance: int | None = None,
    learning_rate: float = 0.1,
    prior: PriorBounds | None = None,
) -> ExperimentConfig:
    budget = choose_budget(ode_budget_traj, lam=budget_lambda)
    if l_contrastive is not None or grad_batch is not None or m_nuisance is not None:
        budget = BudgetChoice(
            l_contrastive=int(
                l_contrastive if l_contrastive is not None else budget.l_contrastive
            ),
            grad_batch=int(grad_batch if grad_batch is not None else budget.grad_batch),
            m_nuisance=int(m_nuisance if m_nuisance is not None else budget.m_nuisance),
        )

    return ExperimentConfig(
        n_steps=int(n_steps),
        dt=float(dt),
        n_substeps=int(n_substeps),
        ode_budget_traj=int(ode_budget_traj),
        budget_lambda=float(budget_lambda),
        budget=budget,
        prior=prior if prior is not None else PriorBounds(),
        learning_rate=float(learning_rate),
    )
