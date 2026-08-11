# Speaker notes

Prepared material runs about 25 minutes over 18 slides. Frame numbers match
`slides.pdf`.

**Split.** Frames 2–7 and 14–18 are the experimental-design half (Arno): about
10:20 of theory plus 7:00 of results. Frames 8–13 are the Julia half (Sebastian):
about 7:30 as it stands, and the section still to be finetuned.

## Pacing

| # | Frame | Target |
|---|---|---|
| 1 | Title | 0:20 |
| 2 | Notation and goal | 1:30 |
| 3 | Fed-batch bioreactor | 1:30 |
| 4 | Scoring a design | 1:30 |
| 5 | Why that is expensive | 1:30 |
| 6 | From static schedules to policies | 2:30 |
| 7 | What one gradient step costs | 1:30 |
| 8 | The stack | 1:00 |
| 9 | The stack, honestly | 1:30 |
| 10 | The policy | 1:00 |
| 11 | The objective, as written | 1:30 |
| 12 | The same inner loop in JAX | 1:30 |
| 13 | Matched-workload benchmark | 1:00 |
| 14 | What the learned policy does | 1:30 |
| 15 | Sharper posteriors | 1:30 |
| 16 | Two runs, two experiments | 1:30 |
| 17 | DC motor | 2:00 |
| 18 | Code and paper | 0:30 |

Nothing to cut at present; if anything, the deck is short.

## Per frame

**2 — Notation and goal.** Assume no experimental-design background. Introduce
theta, u, y one at a time and say explicitly that "design" means the settings of
the apparatus, not a model or a network architecture — that is the word this
audience will mis-hear. Bayes' rule is on the slide only so that "posterior" is
concretely defined; do not derive anything. Land the last line: we are choosing u
*before* any data exist, so the whole problem is about data we have not seen yet.
That is what separates design from estimation.

**3 — Bioreactor.** Walk the picture, not the equations: the pump is the knob, the
probe is the measurement, the two kinetic constants are what we want. Say the ODE
is three states and one algebraic growth law, then move on. Fourteen decisions,
one per hour; a wasted hour cannot be replayed. Mention in passing that the noise
scale and the initial biomass are unknown too, and that we come back to it.

**4 — Scoring a design.** Two beats only. Entropy, in words: how spread out a
distribution is, low when the posterior is sharp. Then the one real move: while
designing we have no observations yet, so we average over every observation the
experiment could produce. Two things to keep spoken rather than on the slide: for
a Gaussian posterior this is classical D-optimality, which is where the
Fisher-information baseline in the result figures comes from; and the small print at the
bottom is our targeted variant — only some parameters are of interest, the rest
are averaged out, and the paper has that math.

**5 — Why that is expensive.** Emphasise that each likelihood is a full ODE
solve, so the inner sums are simulation, not arithmetic. There are no citations on
the slides at all; if asked, this nested estimator is Rainforth et al. (2018), it
is biased, and the bias falls off like 1/L, so more outer samples do not fix it.

**6 — Schedules to policies.** The longest frame in the talk and the hinge of the
first half; three beats. Static: one big optimisation up front. Adaptive: the
honest statement is the alternating chain of min and expectation — say "dynamic
programming" once and do not unroll it, and say that standard practice
approximates that chain *inside* the running experiment (measure, update the
posterior, re-optimise) which is the setup for the DC motor at the end. Then the
move: optimise a rule instead of the inputs, let a network be the rule, and you
are in territory this audience knows — policy gradient methods. The amortisation
point is the last clause: all the compute is offline, and in the lab a step is one
forward pass. Do not derive the equivalence — the slide just asserts that the best
policy is as good as V, and the proof is in the paper. If
someone asks about REINFORCE: the likelihood is explicit and the simulator
differentiable, so we take pathwise gradients, not score-function ones. Nothing in
the deck is cited any more: amortized design is DAD (Foster et al.) and iDAD
(Ivanova et al.), and the paper carries the full list.

**7 — Cost.** Three beats, one question each; do not read the slide as prose.
*How many ODE solves does a decent estimate take?* — L decoys plus M nuisance
draws plus the rollout, per episode, times fifty thousand episodes: 5e7
trajectories, 3.5e10 RK4 steps. *And we need the gradient, not the value* —
reverse mode through all of it, a thousand times. *So we need* — GPU, one
compiled program, AD through the solver, which is exactly the next slide. Two
things worth saying that are not on the slide: including the true parameter among
the decoys is what makes sPCE a bound rather than an estimate, so it cannot exceed
log(L+1) nats; and the noise is reparameterised, which is what makes the rollout
differentiable.

**8 — The stack.** Lux's explicit parameters are the reason the loss is a pure
function and traces cleanly. Say "hand-written fixed-step RK4" — it is not a
SciML solver call, though SimpleDiffEq's RK4 is a drop-in.

**9 — The stack, honestly.** The frame this audience will remember, so tell it
straight and without grievance. Pros first, briefly. Then the three costs, and let
the third one breathe: the gradient was silently wrong, the fix required a Julia
version we were not on, and the upgrade traded that bug for a different one. The
lesson is the alert, not the anecdote — pin versions, and diff the gradient
against a small unfused reference before spending GPU hours. If asked which
versions: the pins are in `Project.toml` and `benchmark_handoff.md`.

**10 — The policy.** Do not call the attention causal or masked: the buffer is
zero in slots that have not been measured, which is what prevents future
information from entering. The positional encoding is the deviation from DAD.

**11 — The objective.** The point for this audience: loops and in-place writes,
no functional rewrite, no vmap; hypotheses are just an extra array axis. Only
the last line mentions the compiler.

**12 — JAX.** Be fair. Both lower to StableHLO and XLA; the differences are in
how you express the loop and where rematerialisation is decided. Do not claim
anything here is impossible in JAX.

**13 — Benchmark.** State that the numbers are not ready and why: the two
harnesses do not yet time the same region, so any current comparison would be
measuring the harness. Name the contract, then move on. Do not quote the draft
numbers from `benchmark_handoff.md`.

**14 — Monod rollouts.** No text on the slide by design; talk the figure. Left:
substrate. Middle: biomass. Right, the one that matters: the feed. The two static
designs are single step functions — one shape for every experiment. The adaptive
policy is a family of shapes, one per rollout, and it starves the reactor early so
the substrate curve becomes sensitive to the kinetics before it feeds hard. Twenty
rollouts per strategy, shared prior draws, if anyone asks.

**15 — Posteriors.** Red is the adaptive policy, green and blue the static designs,
the cross is truth: the red cloud is tighter. Read one row of the table, not three.
Methodology if asked: sPCE at L = M = 5000 over 1000 held-out episodes, paired
differences +0.23 and +0.34 nats (t = 5.3 and 7.5); RMSE over 5000 trials. Also
volunteer the failures — ten of those 5000 adaptive trials go wrong when an early
misleading reading makes the policy stop feeding, and they are inside every number
on the slide.

**16 — Two runs, two experiments.** Set the scene, because the slide does not: same
reactor, but now high substrate can inhibit growth, and the only thing we want to
learn is how strong that inhibition is (alpha in the legend). Then the point: the
opening probe is identical in both runs — push substrate up to where inhibition
would show — and the two designs separate only afterwards, at hour 10 versus 11.
The timing of the switch is the adaptation. The PK model in the paper makes the
same point about parameters we do *not* care about: how fast the drug clears
changes when you should dose to learn absorption.

**17 — DC motor.** The one job of this slide: amortization is what makes adaptive
design real-time. Point at the two clouds. Adaptive (BIM) re-optimizes online and
its per-step time climbs into the 10 ms budget, crossing it around step 5 — and
that baseline is already generous (myopic, Dirac posterior, one-dimensional grid
search). Adaptive (sPCE) is flat at about 24 microseconds, two orders of magnitude
under the deadline, because the optimization happened offline. Deployment is plain
Lux on one CPU core: no Reactant, no XLA, no GPU. If accuracy is asked for as well:
2.96 vs 2.56 nats, and lower RMSE on both k and J.

## Likely questions

- *What is the bound, exactly?* Sequential prior contrastive estimation (Foster
  et al., 2021), with our targeted extension for nuisance parameters. It is
  stated on frame 7 in words only; the derivation is in the paper.
- *Why not reinforcement learning?* The likelihood is explicit and the simulator
  is differentiable, so pathwise gradients are available and lower variance.
  Score-function methods are what you need for discrete or non-differentiable
  designs.
- *Why not a variational posterior (Barber–Agakov, flows)?* sPCE needs no
  posterior approximation, and with an explicit likelihood the contrastive bound
  is the cheaper option. A flow would also give you inference at deployment.
- *How would a prior shift be handled?* Retraining, today. Training on a wider
  prior is the practical answer, at the cost of a looser policy.
- *Compile time?* Minutes for the traced objective, once per shape. It is a real
  cost, and one of the two numbers the benchmark slide keeps separate.
- *Does the policy see the parameters?* No. Only past inputs and measurements.

**18 — Code and paper.** Leave it up while taking questions. The QR is the
repository; the arXiv number is the paper with the proofs, the references and the
two examples that are not in the talk.

## Still open

**Julia half (8–13).** Frame 13's results table is deliberately empty; the gate for
filling it is `benchmark_handoff.md`. Frame 9's version-chase anecdote carries no
package versions — add them if wanted. A "to be continued" slide on DiffEqReactant
would sit naturally after 13, and would also lengthen a section that is currently
the shorter of the two.

**Design half, optional.** Two things I would still consider, both cheap:

1. An opening slide before frame 2 on why the schedule is worth optimizing at all
   (a reactor run is a day, bench time is booked) and that it is chosen before any
   data exist. The talk currently starts at notation.
2. One slide on the classical route — linearize, Fisher information, D-optimality —
   because "BIM" labels a curve in all three result figures and is never explained,
   and because it pays off the Gaussian remark that is now spoken-only on frame 4.

The DC motor slide also has no numbers on it: 2.96 vs 2.56 nats, 24 microseconds vs
7.4 ms per step. Say them out loud, or put the small table back.
