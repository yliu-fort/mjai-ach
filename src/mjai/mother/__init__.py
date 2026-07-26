"""The generative mother G_phi (AGENTS.md D12, D18; 研究计划 §3.1).

Layer: floor-adjacent, sibling of ``mjai.games`` / ``mjai.agents``. Imports
``mjai.utils`` only.

What lives here: the in-house RealNVP-style coupling flow used in Phases A/B
(output dim 12-48, exact change-of-variables logdet, so entropy and KL are
exact and differentiable) and, from Phase C, the shared-trunk + z-conditioned
low-rank modulation hypernet. Written in-house rather than taken from
``nflows`` per D18.

Sealed off from the evaluation side by the layering: a mother never sees a
certificate, an atlas entry, or an opponent pool.
"""
