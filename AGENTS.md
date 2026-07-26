# AGENTS.md — Governance for humans and AI

This document is the contract for everyone (and every agent) working in `mjai-ach`.
It is written **before** code so that code conforms to it, not the reverse.
If a change would violate a rule here, either change the code or amend this file
through review — never silently break a rule.

The project is an IMPALA-style PPO/ACH + league-play research pipeline for
imperfect-information games. Phase 1 (this repo, now) targets tabular + small-MLP
on a home CPU+GPU across 8 small games. Phases 2 (4-player Mahjong) and 3
(128-core + multi-GPU on SLURM) are config + tuning only *for the base stack*.

**A second research programme now shares this repo**: pACH — a generative mother
network whose push-forward distribution is the population, trained by population
self-play with no historical opponents. Its pre-registration and single source of
truth is `Generative-ach.md` (research plan v3); the base ACH stack is its
engineering basis and its evaluation-side toolkit. That programme adds code, so
"all code is written in Phase 1" no longer holds; D12 below governs where it goes.

> **Two D# namespaces.** The `D#` rows in this file are *repository governance*.
> `Generative-ach.md` carries its own `D1–D18` table of *research design*
> decisions. They do not share a namespace — always cite the document:
> `AGENTS.md D12` vs `研究计划 D5`.

---

## 1. Locked decisions (do not relitigate without amending this file)

| # | Decision | Value |
|---|---|---|
| D1 | Tensor backend | PyTorch (no JAX — no Windows-GPU support) |
| D2 | Executor | Ray; core logic separated from Ray; SLURM launcher shipped in Phase 1 |
| D3 | Multi-agent core | Built in-house (OpenSpiel's `PPOAgent` is single-agent only) |
| D4 | ACH actor loss | Paper-faithful ACH (Fu et al. ICLR 2022, OpenReview DTXZqTNV5nW, Algorithm 2): logit-space policy gradient + advantage-sign-dependent one-sided logit gate (l_th) + ratio gate (vacuous under synchronous single-thread) + entropy regularizer + critic value loss; **no PPO clipped-surrogate in the ACH policy loss** (`theta=1` short-circuits the PPO term — it is never built, see D11) |
| D5 | Tabular | From-scratch dict-backed policy + value, same `Trainer` interface as the NN |
| D6 | GPU | Training defaults to GPU; `torch.cuda.is_available()` asserted unless `--cpu` or `MJAI_CPU=1`. **No silent degradation.** |
| D7 | Environment | Python `==3.12.*` fixed; `uv` + `pyproject.toml` + `uv.lock` |
| D8 | Games | All 7: BRPS, Goofspiel-5 II, Liar's-Dice-1, Oshi-Zumo, Leduc, Kuhn, Tic-Tac-Toe |
| D9 | Logging | **TensorBoard only.** One `SummaryWriter` per run. No W&B, no JSON loggers. |
| D10 | Play CLI | `mjai-play` console-script; menu-driven; loads any saved policy vs human or auto |
| D11 | NN policy-improvement interface | **One** update rule, `NNActorCriticUpdate`, parameterized by `theta`: policy loss = `(1-theta)*L_ppo_clip + theta*L_ach`. `algo: ppo`/`ach` are pinned aliases for `theta=0`/`1`; `algo: theta` sweeps it. Everything outside the policy term (optimizer, advantage treatment, epochs per batch, grad clipping) is ONE shared knob-driven scaffold whose defaults follow the ACH protocol **at every theta**, so PPO-vs-ACH varies only `theta` unless a knob is set deliberately. Knobs the paper contradicts emit `ACHFidelityWarning` when `theta > 0`. Amends audit B10 (which had removed the interpolation); the merge is licensed by `tests/unit/data/nn_updates_golden.json`, a pre-merge fixture the unified rule must still reproduce exactly. |

### 1.1 pACH-phase decisions (added 2026-07-26)

Each row below was ratified against `Generative-ach.md` before any code was written.

| # | Decision | Value |
|---|---|---|
| D12 | pACH lives here | The generative-mother programme shares this repo. Five new subpackages slot into the §2 layering: `mjai.seqform`, `mjai.atlas`, `mjai.mother`, `mjai.pach`, `mjai.certs`. The research plan's flat `games/ atlas/ mother/ pach/ certs/ analysis/` map onto them. There is **no `basis/` package** — the "basis" *is* `mjai.algos` + `mjai.eval`; the hand-off artifact is `docs/basis_audit.md`. |
| D13 | Games | D8's seven **plus `kuhn3`** (`kuhn_poker(players=3)`) = 8. `kuhn3` is registered in full per §4 (config + renderer + input parser + CLI smoke test), which required generalizing `MatchRunner` from a hard-coded 2 seats to n. Rationale: 3p Kuhn is the Phase-B decision gate; carving out a "research-only game" exemption would have split the game registry in two. **`kuhn3` is deliberately NOT in the Phase-1 experiment matrix** — there are no `configs/exp/kuhn3_*.yaml`, and the one-click notebook stays at 7 games / 28 cells, because the base-stack study is a 2p study and ACH's guarantee does not extend to n ≥ 3. Do not "fix" the missing configs. |
| D14 | Exact-eval parity | The research plan asks for "bit-for-bit" three-way agreement; that is **unattainable and the claim is amended here**. Parity compares **evaluators on one fixed policy**, never solvers against each other. For n ≥ 3, CFR is *not* a Nash reference (multiplayer CFR has no such guarantee) — it is only a convenient non-trivial test policy. Tolerances, measured 2026-07-26 over random / uniform / near-equilibrium policies on Kuhn, 3p Kuhn and Leduc (`tests/unit/test_seqform_parity.py`): **`mjai.seqform` vs OpenSpiel — 0 or 1 ulp**, held at ≤1e-12 absolute; **either vs the base stack — ≤1.5e-14 relative**, held at ≤1e-11. Our own route is additionally required to be **bit-reproducible across processes**, which the C++ backend is not. |
| D14a | Exact eval carries float64 | `Policy.action_logits_batch` returns **float64**, and `MLPSharedActorCritic`'s override widens at the numpy boundary while keeping the forward in float32. Measuring D14 is what found this: the base stack sat 6e-10 to 1.8e-8 relative from an independent implementation, and it was *not* backend noise — its two best-response backends agreed with each other exactly. It was the float32 return capping the exact evaluator no matter which solver ran under it. Widening the handoff invents no precision: a tabular policy's logits are Python floats and are now **bit-identical** between the fast and reference routes, while an NN's residual ~1e-8 is its own float32 weights, a property of the model rather than of the metric. Do not narrow it back; `test_exact_eval_receives_float64_logits` guards the contract next to the tolerance it justifies. Exact eval values moved by ≤1.8e-8 relative on 2026-07-26 and are not bit-comparable across that date — no committed result changes in any digit it reports, so nothing was regenerated. |
| D15 | Simplex validation | Every evaluator we write validates that each policy row is a probability distribution and raises otherwise. Measured 2026-07-26: OpenSpiel's `nash_conv` silently accepts p > 1 and returns a **negative** NashConv (the Kuhn α-family at α = 0.4 makes 3α = 1.2 and yields −6.7e-2). Silent acceptance of an invalid policy is exactly the failure mode §11 forbids. |
| D16 | Average-policy anchor | ACH's `O(T^-1/2)` guarantee is about the **average** strategy, but `docs/reproduce_report.md` records only the current policy π = softmax(y). The pipeline sanity anchor the research plan §5.0(7) asks for therefore needs a running-average-policy exploitability tracker. Enabled on Kuhn only; off elsewhere. |
| D17 | KL trust-region anchor is compliant | pACH's proximal term `KL(ρ_φ ‖ ρ_φt)` holds a copy of the previous mother φ_t. Ratified reading: φ_t is PPO's `π_old` — frozen only across one meta-iteration's inner loop, and **never used as an opponent**. It therefore does not breach the research plan's no-history hard constraint (which binds opponents: no snapshots, no EMA mother, no opponent pool). |
| D18 | pACH device & dependencies | Phase A/B is 12–48 dimensional; GPU is a net loss. pACH configs carry an **explicit `device` field** (default `cpu` for Phase A/B) rather than routing around D6's assert — the device is a property of the config, knowable before the run, never a silent degradation. The mother's RealNVP coupling flow is written in-house (≈100 lines, exact logdet) instead of taking `nflows`, which mypy-strict cannot type. `sympy` is added to the dev extra for the Step-0 symbolic derivations. |
| D19 | Numeric precision | Oracle track and all certificates are float64; the MC track runs float32 and casts to float64 before any certificate. Tolerances must separate *algorithmic* error from *platform ulp* — `tests/unit/test_algos_nn_theta.py` is a standing reminder that float noise differs across macOS/CPU torch builds. |

---

## 2. Layering rule (enforced by import-linter)

Layers, high → low. A layer may import only the layers **below** it:

```
mjai.cli        ← leaf; nothing in the repo imports it
mjai.pipeline   ← the ONLY place training meets evaluation (see below)
mjai.certs
mjai.eval
mjai.atlas
mjai.league
mjai.algos      ↤ siblings (no import direction between them)
mjai.pach       ↤
mjai.config
mjai.seqform
mjai.games      ↤ siblings
mjai.agents     ↤
mjai.mother     ↤
mjai.utils      ← floor; imports nothing internal
```

Additional constraints enforced separately:

- **`mjai.league` must not import concrete algos.** It imports only the abstract
  controller/Trainer interfaces in `mjai.algos.controller`. Forbidden:
  `mjai.algos.ppo`, `mjai.algos.ach`, `mjai.algos.trainer`, `mjai.algos.baselines`.
  Rationale: league provides the self-play *controller*; it must not depend on a
  specific update rule or on the Trainer class that consumes controllers
  (otherwise a cycle forms).
- **The pACH training loop is sealed off from the evaluation side.** `mjai.pach`
  and `mjai.mother` sit *below* `mjai.atlas`, `mjai.eval`, `mjai.certs` and
  `mjai.league`, so the layering alone makes it impossible for the training loop
  to consult ground truth, a certificate, an exploitability oracle, or an
  opponent pool. This is not a style preference: it is the research plan's核心
  central hard constraint (training may only ever see fresh samples of the current
  population and its own terminal payoffs), turned into a machine-checked
  contract. A second, redundant `forbidden` contract states the same thing
  explicitly so the intent survives a future re-ordering of the layers.
  **Consequence:** the assembly point where a training run periodically emits
  NashConv/BR-gap curves is `mjai.pipeline`, which sits above both. If you find
  yourself wanting to import `certs` from `pach`, the wiring belongs in the
  runner instead.
- **`mjai.atlas` must not import `mjai.eval`.** The Step-0 ground truth and the
  base stack's evaluators are two of the three legs of the D14 parity check; a
  parity check between an implementation and itself proves nothing.
- **`mjai.cli` is a leaf.** No module under `mjai.*` may import `mjai.cli`.
  (The console-script entry point imports it from outside the package — that's fine.)
- **Tests may import anything**, but must not be imported by anything.

Violations fail pre-commit and CI.

---

## 3. File rules

1. **Every source file is under 500 lines and has a single responsibility.**
   Enforced by an AST-based guard in pre-commit. If a file is growing past ~400
   lines, split it along the next natural seam before it hits the cap.
2. **Composition over inheritance.** Base classes define *interfaces* (abstract
   methods, Protocols); behavior is composed from small components. Deep
   inheritance hierarchies are a smell.
3. **Subclasses interact through the base-class interface only.** Callers must
   not type-check or downcast. If you find yourself writing `isinstance(x, Foo)`,
   the interface is probably wrong — extend the base instead.
4. **High cohesion, low coupling.** A module changes for one reason. If a change
   touches more than two unrelated modules, the boundary is wrong — propose a
   refactor rather than spreading the change.
5. **No god objects, no catch-all utils.** `mjai.utils` holds only leaf helpers
   (seeds, gpu-assert, logging setup, ckpt I/O). If a util starts to know about
   policies or games, it belongs elsewhere.
6. **Naming**: `snake_case` for modules/functions/variables, `PascalCase` for
   classes, `UPPER_SNAKE` for module constants. Files are `snake_case.py`.
   One public class per file when practical; otherwise group closely-related
   small classes.

---

## 4. How to add things

### Add a game
1. Add `configs/games/<game>.yaml` (game string, params, dynamics flags).
2. If OpenSpiel's `information_state_tensor` is not directly usable, add a thin
   adapter in `games/adapters/<game>.py`. Most games need no adapter.
3. Add `cli/renderers/<game>.py` and `cli/input_parsers/<game>.py`. **Required** —
   the CLI smoke test and import checker reject a game missing either.
4. Add a unit test in `tests/unit/` covering info-state shape, action masking,
   and a short rollout.
5. Run `uv run pre-commit run --all-files` — all gates must pass.

**Simultaneous-move games** (BRPS, Goofspiel, Oshi-Zumo): the human's input is
**blind** — prompt on own private info only, never reveal the opponent's
simultaneous choice. Loaded policies must be exercised on the true game tree.

### Add an algorithm
1. Subclass `mjai.algos.update_rule.UpdateRule`. Implement `step(batch) -> (loss, stats)`.
2. Do **not** edit `Trainer`, the pipeline, or the league. The new rule composes
   into the existing Trainer by configuration.
3. Add a unit test on a tiny game comparing against CFR / exact Nash where possible.

**Changing the NN actor-critic instead of adding a rule** (D11): the policy term
is the only thing `theta` interpolates; a PPO best practice that lives outside it
belongs in `AlgoConfig` as a knob, defaulting to the ACH-protocol value, and must
be added to `_warn_if_ach_incompatible` if the paper contradicts it. Any change
to `nn_losses.py`/`nn_updates.py` must keep
`uv run python tools/gen_nn_golden.py --check` passing — that check is what
protects the paper reproduction from collateral damage.

### Add an experiment
1. Add `configs/exp/<name>.yaml` selecting game + algo + self-play mode (mirror|league).
2. Run via `uv run python scripts/train.py --config configs/exp/<name>.yaml`.
3. Eval via `scripts/evaluate.py` on the produced run directory.

### Add a metric to eval
1. Add `eval/<metric>.py` exposing a single function taking a run directory.
2. Register it in the eval config so the one-click notebook and `evaluate.py` pick it up.

---

## 5. Test contract

- **Every public module has unit tests** under `tests/unit/`.
- **New games get a CLI smoke test** (launch, pick game, auto-run one match
  between two fixed random policies, assert clean exit + correct returns).
- **Integration smoke tests** on BRPS + Goofspiel-5 run on every push.
- Tests are deterministic: seed everything; no wall-clock or RNG-from-system-time.
- Tests do not require GPU unless explicitly marked `@pytest.mark.gpu`; the fast
  unit suite (run at pre-commit) is CPU-only and <20s.
- A test that fails intermittently is a bug. Fix it or mark it `xfail` with a reason.

---

## 6. Logging contract (TensorBoard only)

- Exactly **one** `torch.utils.tensorboard.SummaryWriter` per run, created by the
  runner and passed down. No module creates its own writer.
- Scalars: loss components, entropy, value-R², grad-norm, exploitability/NashConv
  curves, weight-delta between checkpoints.
- Hparams + full config dumped once at run start via `add_hparams`.
- Artifacts (payoff matrices, plots) saved as files in the run directory and
  referenced from the notebook — do **not** push large arrays through TensorBoard.

---

## 7. Notebook contract

Three families, all generated by a builder under `tools/` (never hand-edited —
edit the builder and regenerate):

| Family | Builder | What it answers |
|---|---|---|
| `phase1_one_click.ipynb` | `tools/build_notebook.py` | the whole Phase-1 matrix |
| `ab_<game>.ipynb` | `tools/build_league_notebooks.py` | mirror vs league (F2) |
| `theta_<game>.ipynb` | `tools/build_theta_notebooks.py` | PPO⟷ACH theta scan (D11) |
| `ppo_vs_ach_<game>.ipynb`, `sgd_vs_adam_<game>.ipynb` | `tools/build_ab_factor_notebooks.py` | single-factor A/B isolating one axis (algorithm or optimizer) on liar's dice, mirror — backed by `tools/ab_factor_probe.py`. Adds one arm per side; `ppo_vs_ach` separates "policy term alone" (theta=0 on the ACH scaffold) from "each algorithm as its authors intended" (tuned-PPO config); `sgd_vs_adam` holds ACH fixed and varies the optimizer (the Adam arms emit the expected `ACHFidelityWarning`). |

`notebooks/phase1_one_click.ipynb` is the single human-facing entry point:
- One parameterized cell at the top (game, algo, mode, or "run all 28 cells").
- Top-to-bottom readable; no hidden state; rerunnable from a clean kernel.
- Trains, then runs the full eval toolkit, then renders the payoff matrix /
  forgetting / non-transitivity plots.
- It imports from `mjai.*` — it does **not** reimplement logic.

---

## 8. Performance & memory discipline

- **Profile before optimizing.** `scripts/profile_pipeline.py` gives a per-component
  breakdown (cProfile for CPU rollout/learner/league; `torch.profiler` for GPU
  fwd/bwd + dataloader). No micro-optimization lands without a profile showing it matters.
- **No memory leaks.** Checkpoint pools use weak refs / TTL bounding so stale
  checkpoints can be GC'd; the integration smoke test takes a `tracemalloc`
  snapshot and fails on unbounded growth over a fixed step window.
- **Ray object store**: be explicit about what gets put in the store (weights
  only, by default); never put unbounded histories there.

---

## 9. Configuration discipline

- All hyperparameters live in YAML under `configs/`. No magic numbers in code.
- Configs load into frozen dataclasses (`mjai.config`) via `cattrs`. Unknown keys
  error loudly — no silent ignores.
- A run's full config is dumped into its run directory at start, so a result is
  always traceable to the exact config that produced it.

---

## 10. Commit & PR discipline

- One logical change per commit. Commit message format: `<area>: <imperative summary>`
  (e.g. `algos: add ACH update rule`, `games: add goofspiel5 config`).
- Pre-commit must pass on staged changes. Pre-push runs the full suite.
- Never push directly to `main`. Feature branches → PR → review.
- A PR that adds a public module must add its unit tests in the same PR.

---

## 11. AI-agent operating rules

When acting as an AI agent in this repo:

- **Read this file first.** If a requested change violates a rule here, say so and
  propose the compliant alternative; do not silently comply.
- **Follow the layering.** If an import would point upward, stop and redesign.
- **Keep files under 500 lines.** Split before merging, not after.
- **Never introduce a silent fallback** (especially for GPU). Fail loudly.
- **One concern per change.** Don't bundle a refactor with a feature.
- **Cite the rule** you're following when a decision is non-obvious.
- **Run the gates** (`uv run pre-commit run --all-files`) before declaring done.
  Report failures honestly; don't claim success without the output.
