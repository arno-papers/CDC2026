"""JAX implementation of Monod bioreactor adaptive sPCE design."""

from .model import (
    N_STEPS,
    DT,
    N_SUBSTEPS,
    ACTION_HI,
    dynamics,
    rk4_step,
    integrate,
    sample_theta_full,
    sample_theta_obs,
    allocate_budget,
)
from .batched import batched_spce_loss, blocked_batched_spce_loss
from .policy import init_params, policy_forward
from .training import single_episode_loss, batch_loss, train  # noqa: F401
