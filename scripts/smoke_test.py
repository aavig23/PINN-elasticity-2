"""Minimal end-to-end PINN smoke test (not a training experiment).

    python scripts/smoke_test.py                 # both stages, GPU if available
    python scripts/smoke_test.py --device cpu --stages 1

Trains the smoke configs (tiny networks, a few dozen iterations) and writes to
the git-ignored outputs/smoke/<label>/stage<n>/. It checks that the pipeline
runs on the chosen device: forward pass, autograd derivatives, residuals,
losses, Adam and L-BFGS steps, checkpoints and diagnostics. The loss values
say nothing about solution quality.

Exit code 0 = every stage completed with a finite loss, 1 = otherwise.
"""

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="cpu, cuda, ... (default: cuda if available, else cpu)")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    parser.add_argument("--label", default="smoke_test", help="subfolder under the output root")
    parser.add_argument("--out-root", default=str(ROOT / "outputs" / "smoke"), help="default: outputs/smoke/")
    args = parser.parse_args()

    from pinn_elasticity.config import load_config
    from pinn_elasticity.experiment.logger import read_losses
    from pinn_elasticity.training.trainer import Trainer

    ok = True
    for stage in args.stages:
        cfg = load_config(ROOT / "configs" / "smoke" / f"stage{stage}_smoke.yaml")
        out = Path(args.out_root) / args.label / f"stage{stage}"
        start = time.perf_counter()
        trainer = Trainer(cfg, out_dir=out, device=args.device)
        summary = trainer.run()
        seconds = time.perf_counter() - start

        print(f"== Stage {stage} smoke test ==")
        print(f"  device      {summary['device']} ({summary['device_name']}), dtype {summary['dtype']}")
        print(f"  status      {summary['status']} {summary['message']}".rstrip())
        print(f"  iterations  {summary['iterations']}  in {seconds:.1f} s")
        stage_ok = summary["status"] == "completed" and summary["final"] is not None
        if stage_ok:
            stage_ok = math.isfinite(summary["final"]["total"])
            rows = read_losses(out / "losses.csv")
            print(f"  total loss  first {float(rows[0]['total']):.3e} -> final {summary['final']['total']:.3e}")
            print(f"  terms       " + ", ".join(f"{k}={v:.2e}" for k, v in summary["final"]["terms"].items()))
            print(f"  float32 vs float64 residual: relative diff {summary['precision_check']['relative_diff']:.1e}")
            exact = all(v == 0.0 for v in summary["diagnostics"]["constraints"].values())
            print(f"  hard clamp/IC exactly zero: {exact}")
            stage_ok = stage_ok and exact
        print(f"  output      {out}")
        print(f"  RESULT      {'PASS' if stage_ok else 'FAIL'}\n")
        ok = ok and stage_ok

    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
