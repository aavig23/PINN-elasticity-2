"""Run or resume a PINN experiment.

Full run (committed experiment config, clean working tree; writes runs/EXPnnn/<run_id>/):
    python scripts/run.py --config experiments/EXP001_stage1_baseline/config.yaml
    python scripts/run.py --config experiments/EXP001_stage1_baseline/config.yaml --seeds 0 1 2

Smoke run (any config, no Git checks; writes outputs/smoke/<id>/<run_id>/):
    python scripts/run.py --config configs/smoke/stage1_smoke.yaml --smoke

Resume an interrupted run (same commit as when it started):
    python scripts/run.py --resume runs/EXP001/<run_id>

--max-iterations N stops at global iteration N with status "interrupted"
(resume later with --resume).

Exit code: 1 if any run failed or was refused, else 2 if any run was interrupted, else 0.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _report(result) -> None:
    s = result.summary
    print(f"run folder  {result.run_dir}")
    print(f"status      {s['status']} {s['message']}".rstrip())
    print(f"iterations  {s['iterations']}   time {s['training_time_s']} s   device {s['device_name']}")
    if s["final"] is not None:
        print(f"final loss  {s['final']['total']:.4e}")
    print(f"folder size {s['run_folder']['size_mb']} MB")
    for w in result.warnings:
        print(f"WARNING     {w}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--config", help="experiment config (or any config with --smoke)")
    what.add_argument("--resume", help="run folder to resume")
    parser.add_argument("--seeds", type=int, nargs="+", help="one run per seed (default: the config's seed)")
    parser.add_argument("--smoke", action="store_true", help="skip Git checks; write to outputs/smoke/")
    parser.add_argument("--device", default=None, help="cpu, cuda, ... (default: cuda if available)")
    parser.add_argument("--max-iterations", type=int, default=None, help="stop early at this global iteration")
    args = parser.parse_args()
    if args.resume and (args.seeds or args.smoke):
        parser.error("--seeds and --smoke cannot be combined with --resume")

    from pinn_elasticity.config import ConfigError
    from pinn_elasticity.experiment.git_guard import GitGuardError
    from pinn_elasticity.experiment.runner import RunError, resume_run, start_run

    command = [Path(sys.argv[0]).as_posix(), *sys.argv[1:]]
    codes = {"completed": 0, "interrupted": 2}
    try:
        if args.resume:
            results = [resume_run(args.resume, repo_root=ROOT, device=args.device,
                                  max_iterations=args.max_iterations, command=command)]  # fmt: skip
        else:
            results = []
            for seed in args.seeds or [None]:
                results.append(start_run(
                    args.config, repo_root=ROOT, seed=seed, smoke=args.smoke, device=args.device,
                    max_iterations=args.max_iterations, command=command,
                ))  # fmt: skip
                _report(results[-1])
                print()
    except (GitGuardError, ConfigError, RunError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    if args.resume:
        _report(results[0])
    statuses = {r.summary["status"] for r in results}
    if statuses - set(codes):
        return 1  # a failure outranks an interruption
    return max(codes[s] for s in statuses)


if __name__ == "__main__":
    sys.exit(main())
