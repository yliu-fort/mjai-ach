"""Gap-investigation A/B probe arms on Liar's Dice.

Mid-way reproduction showed ours ~0.32 vs paper mean ~0.17 (audit U1-U6).
Three one-factor-at-a-time arms against the paper-faithful baseline (the 8
reproduction seeds serve as the baseline reference):

  lam1 : gae_lambda=1.0 (spec assumption A1; H.3 unspecified)
  rawy : loss_centered_logits=False (literal Algorithm 2 raw logit; spec A3/U1)
  ent3 : entropy_coef=0.03 (the paper's own ACH beta on FHP, p27 Table 6)
  legalmean : centered_mean_legal_only=True (y_bar over legal actions only;
      A5-adjacent — final-checkpoint diagnostic showed illegal-logit drift
      shifting the gate by up to +/-4 on Liar's Dice)

Each arm: seed 11, 2e6 env-steps, eval every 2e5. Usage::

    python tools/gap_probe.py --arm lam1
    python tools/gap_probe.py --summarize
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb_eval import read_many

from mjai.scripts.experiment import run_experiment
from mjai.scripts.reproduce_paper import _load_exp_config

REPO = Path(__file__).resolve().parents[1]
PROBE_ROOT = REPO / "runs" / "gap_probe"

ARMS: dict[str, dict[str, object]] = {
    "lam1": {"gae_lambda": 1.0},
    "rawy": {"loss_centered_logits": False},
    "ent3": {"entropy_coef": 0.03},
    "legalmean": {"centered_mean_legal_only": True},
}


def run_arm(arm: str) -> None:
    out_dir = PROBE_ROOT / arm
    cfg = dataclasses.replace(
        _load_exp_config("liars_dice1"),
        seed=11,
        out_dir=str(out_dir),
        total_env_steps=2_000_000,
        eval_every_env_steps=200_000,
        verbose=False,
        **ARMS[arm],
    )
    run_experiment(cfg)
    (out_dir / "DONE").write_text("ok\n", encoding="utf-8")


def summarize() -> dict[str, object]:
    dirs = sorted(PROBE_ROOT.glob("*/tb"))
    curves = read_many(dirs)
    out: dict[str, object] = {}
    for d, c in curves.items():
        arm = Path(d).parent.name
        out[arm] = {
            "done": (PROBE_ROOT / arm / "DONE").exists(),
            "n_evals": len(c),
            "curve": [[s, round(v, 4)] for s, v in c],
            "final": c[-1][1] if c else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize()))
        return 0
    if not args.arm:
        parser.error("--arm required unless --summarize")
    run_arm(args.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
