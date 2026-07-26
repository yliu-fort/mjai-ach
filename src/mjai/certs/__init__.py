"""Equilibrium certificates and population geometry (AGENTS.md D12; 研究计划 §4).

Layer: above ``mjai.eval`` and ``mjai.atlas`` (it compares measured against
predicted), below ``mjai.pipeline``.

What lives here: NashConv of the population mean, the per-sample best-response
gap distribution, the participation-ratio rank with its excess spectrum against
a null-mother control, principal angles against the Step-0 tangent basis, and
the coverage statistics in the family's own parameter coordinates.

**Honesty clauses that must survive into every figure caption produced here:**

- For n >= 3 the population solution concept is mean-field / population Nash.
  Each generated actor is a best response to the process it actually faces, but
  a single sampled n-tuple is NOT a Nash profile of the n-player game
  (研究计划 §2.2). Never label it as one.
- Where a diversity regularizer is what maintains rank (Phases C/D), rank
  maintenance is not evidence on its own; the evidence is the support
  certificate plus agreement with an independently computed m* (研究计划 §4.4).
"""
