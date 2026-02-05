from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BudgetChoice:
    l_contrastive: int
    grad_batch: int
    m_nuisance: int


def choose_l_and_b(ode_budget_traj: int, lam: float = 1.0) -> tuple[int, int]:
    """Match `common.jl` budget scan.

    Budget: C_traj = B * (L + 3). We choose L, B to minimize
      obj(L) = 1/B + lam/(L+1)^2
    with B = floor(C/(L+3)).
    """
    c = int(ode_budget_traj)
    if c < 4:
        raise ValueError(f"ode_budget_traj must be >= 4, got {c}")

    # Vectorized scan over L = 1..(C-3), inclusive.
    l = np.arange(1, c - 2, dtype=np.int64)
    b = c // (l + 3)
    obj = 1.0 / b + float(lam) / (l + 1) ** 2

    idx = int(obj.argmin())
    return int(l[idx]), int(b[idx])


def choose_m_nuisance(l_contrastive: int) -> int:
    # Julia: min(4096, max(512, 32*(L+1)))
    return int(min(4096, max(512, 32 * (int(l_contrastive) + 1))))


def choose_budget(ode_budget_traj: int, lam: float = 1.0) -> BudgetChoice:
    l, b = choose_l_and_b(ode_budget_traj=ode_budget_traj, lam=lam)
    return BudgetChoice(l_contrastive=l, grad_batch=b, m_nuisance=choose_m_nuisance(l))
