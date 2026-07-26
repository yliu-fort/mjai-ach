"""pACH: the population-level meta-optimizer (AGENTS.md D12; 研究计划 §3).

Layer: sibling of ``mjai.algos``, below ``mjai.league`` / ``mjai.atlas`` /
``mjai.eval`` / ``mjai.certs``. That ordering is the point: the layering alone
makes it impossible for the training loop to consult ground truth, a
certificate, an exploitability oracle, or an opponent pool — the research
plan's central hard constraint, machine-checked (AGENTS.md §2).

What lives here: the sequence-form ridge critic, the optimistic extrapolation
(2*A_t - A_{t-1}, optimizer state only), the entropy temperature schedule and
the KL trust region whose anchor rho_{phi_t} is PPO's ``pi_old`` (D17).

Periodic certificate logging during a run is wired in ``mjai.pipeline``, which
sits above both sides. If you want to import ``mjai.certs`` from here, the
wiring belongs in the runner instead.
"""
