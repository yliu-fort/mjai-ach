"""Step-0 ground truth: the equilibrium atlas (AGENTS.md D12, 研究计划 §5.0).

Layer: above ``mjai.league``, below ``mjai.eval``. **Must not import**
``mjai.eval`` — atlas and the base stack's evaluators are two independent legs
of the D14 parity check, and a parity check between an implementation and
itself proves nothing.

What lives here: the symbolic 2p Kuhn alpha-family and its -1/18 game value,
the QRE homotopy path and its limiting logit equilibrium, the re-derived 3p
Kuhn family with its local dimension m* and tangent basis, the normal Jacobian
spectrum, and the frozen ``atlas.json`` those produce. Consumers read the JSON;
they do not re-run the derivation.
"""
