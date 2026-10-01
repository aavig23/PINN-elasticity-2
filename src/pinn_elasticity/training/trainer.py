"""Training loop: Adam, then L-BFGS, with logging, checkpoints and resume (plan §12).

Iteration numbering is global: Adam iterations 0..n_adam-1, then L-BFGS
iterations n_adam..n_adam+n_lbfgs-1. A logged row holds the losses at the
start of its iteration (before that iteration's update); a final row after
training holds the losses of the trained model.

With an output directory the trainer writes losses.csv, summary.json and
checkpoints/{latest,final}.pt there. Run folders, metadata, field export and
figures are added by later milestones.
"""

import time
from pathlib import Path

import torch

from ..experiment.logger import LossLogger, write_json
from ..models.constraints import build_pinn
from ..physics.domain import Domain
from ..physics.loading import Loading
from ..physics.scaling import Scales
from ..sampling import DIAGNOSTIC_STREAM, sample_interior, sample_points
from ..utils.device import describe_device, resolve_dtype, select_device
from ..utils.seed import seed_everything
from .checkpoint import load_checkpoint, save_checkpoint
from .diagnostics import pde_excluding_corners, precision_check, run_diagnostics
from .losses import active_groups, active_terms, compute_loss_terms, group_values, weighted_total
from .optimizers import make_adam, make_adam_scheduler, make_lbfgs
from .weighting import build_weighting

CHECKPOINT_FORMAT = 1


class NonFiniteLossError(RuntimeError):
    """The loss became NaN or infinite."""


class Trainer:
    def __init__(self, cfg, out_dir: str | Path | None = None, device: str | None = None):
        self.cfg = cfg
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.device = select_device(device)
        self.dtype = resolve_dtype(cfg.precision.dtype)
        self.stage = cfg.experiment.stage
        self.mode = cfg.constraints.mode

        self.scales = Scales.from_config(cfg)
        self.domain = Domain.from_config(cfg, self.scales)
        self.loading = Loading(self.scales, self.stage, cfg.time.t_r_hat)

        seed_everything(cfg.experiment.seed)
        self.model = build_pinn(cfg, self.domain).to(device=self.device, dtype=self.dtype)
        self.groups = active_groups(self.stage, self.mode)
        self.terms = active_terms(self.stage, self.mode)
        self.weighting = build_weighting(cfg.loss, self.groups)

        opt = cfg.optimizer
        self.adam = make_adam(self.model.parameters(), opt.adam_lr)
        self.scheduler = make_adam_scheduler(self.adam, opt.adam_lr, opt.adam_lr_final, opt.adam_iterations)
        self.lbfgs = None

        self.phase = "adam"  # "adam" -> "lbfgs" -> "done"
        self.phase_iter = 0
        self.elapsed = 0.0
        self.precision = None
        self.resume_count = 0
        self._resumed = False
        self._points_cache: dict = {}
        self.logger: LossLogger | None = None
        if opt.adam_iterations == 0:
            self._start_lbfgs()

    # Bookkeeping --------------------------------------------------------------
    @property
    def global_iteration(self) -> int:
        n_adam = self.cfg.optimizer.adam_iterations
        if self.phase == "adam":
            return self.phase_iter
        if self.phase == "lbfgs":
            return n_adam + self.phase_iter
        return n_adam + self.cfg.optimizer.lbfgs_iterations

    def _columns(self) -> list[str]:
        return [
            "iteration", "phase", "phase_iteration", "wall_time_s", "lr", "total",
            *self.terms, *(f"w_{g}" for g in self.groups), "pde_excl_corners",
        ]  # fmt: skip

    def _wall(self) -> float:
        return self._elapsed_base + (time.perf_counter() - self._t0)

    def _points(self, index: int):
        if index not in self._points_cache:
            self._points_cache = {
                index: sample_points(
                    self.domain, self.cfg.sampling, constraints_mode=self.mode, seed=self.cfg.experiment.seed,
                    index=index, dtype=self.dtype, device=self.device,
                )  # fmt: skip
            }
        return self._points_cache[index]

    def _adam_index(self, it: int) -> int:
        return it // self.cfg.sampling.resample_every

    def _lbfgs_index(self) -> int:
        # L-BFGS keeps the last Adam point set fixed (it needs a deterministic objective).
        return self._adam_index(max(self.cfg.optimizer.adam_iterations - 1, 0))

    def _evaluate(self, points, iteration: int):
        terms, interior = compute_loss_terms(self.model, points, self.scales, self.loading, self.stage, self.mode)
        weights = self.weighting(terms, iteration)
        return terms, weights, weighted_total(terms, weights), interior

    def _check_finite(self, total: torch.Tensor) -> None:
        if not bool(torch.isfinite(total)):
            raise NonFiniteLossError(f"non-finite loss ({float(total.detach())}) at iteration {self.global_iteration}")

    def _log(self, phase, terms, weights, total, interior, lr) -> None:
        row = {
            "iteration": self.global_iteration,
            "phase": phase,
            "phase_iteration": self.phase_iter,
            "wall_time_s": round(self._wall(), 3),
            "lr": lr,
            "total": float(total.detach()),
            **{t: float(terms[t].detach()) for t in self.terms},
            **{f"w_{g}": weights[g] for g in self.groups},
            "pde_excl_corners": pde_excluding_corners(
                interior, self.cfg.loss.corner_exclusion_radius, self.domain.z_max
            ),
        }
        self.logger.log(row)

    # Checkpoints ----------------------------------------------------------------
    def _state(self) -> dict:
        return {
            "format": CHECKPOINT_FORMAT,
            "config_hash": self.cfg.config_hash,
            "phase": self.phase,
            "phase_iter": self.phase_iter,
            "elapsed": self.elapsed,
            "resume_count": self.resume_count,
            "precision": self.precision,
            "model": self.model.state_dict(),
            "adam": self.adam.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "lbfgs": None if self.lbfgs is None else self.lbfgs.state_dict(),
        }

    def _save(self, name: str) -> None:
        if self.out_dir is None:
            return
        self.elapsed = self._wall()
        save_checkpoint(self._state(), self.out_dir / "checkpoints" / name)

    def _maybe_checkpoint(self) -> None:
        if self.global_iteration % self.cfg.training.checkpoint_every == 0:
            self._save("latest.pt")

    def resume(self, checkpoint_path: str | Path) -> "Trainer":
        """Restore the state saved in a checkpoint; the next run() continues from it."""
        state = load_checkpoint(checkpoint_path, map_location=self.device)
        if state["format"] != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format {state['format']}")
        if state["config_hash"] != self.cfg.config_hash:
            raise ValueError("checkpoint was written with a different configuration (config hash mismatch)")
        self.model.load_state_dict(state["model"])
        self.adam.load_state_dict(state["adam"])
        self.scheduler.load_state_dict(state["scheduler"])
        if state["lbfgs"] is not None:
            self._start_lbfgs()
            self.lbfgs.load_state_dict(state["lbfgs"])
        self.phase, self.phase_iter = state["phase"], state["phase_iter"]
        self.elapsed, self.precision = state["elapsed"], state["precision"]
        self.resume_count = state["resume_count"] + 1
        self._resumed = True
        return self

    # Phases -----------------------------------------------------------------------
    def _start_lbfgs(self) -> None:
        opt = self.cfg.optimizer
        self.lbfgs = make_lbfgs(self.model.parameters(), opt.lbfgs_history, opt.lbfgs_line_search)
        self.phase, self.phase_iter = "lbfgs", 0

    def _budget_reached(self, max_iterations: int | None) -> bool:
        return max_iterations is not None and self.global_iteration >= max_iterations

    def _run_adam(self, max_iterations) -> bool:
        """Run the Adam phase; False if stopped early by ``max_iterations``."""
        while self.phase_iter < self.cfg.optimizer.adam_iterations:
            if self._budget_reached(max_iterations):
                return False
            it = self.phase_iter
            points = self._points(self._adam_index(it))
            self.adam.zero_grad(set_to_none=True)
            terms, weights, total, interior = self._evaluate(points, self.global_iteration)
            self._check_finite(total)
            if it % self.cfg.training.log_every == 0:
                self._log("adam", terms, weights, total, interior, lr=self.adam.param_groups[0]["lr"])
            total.backward()
            self.adam.step()
            self.scheduler.step()
            self.phase_iter += 1
            self._maybe_checkpoint()
        self._start_lbfgs()
        self._save("latest.pt")
        return True

    def _run_lbfgs(self, max_iterations) -> bool:
        """Run the L-BFGS phase; False if stopped early by ``max_iterations``."""
        points = self._points(self._lbfgs_index())
        while self.phase_iter < self.cfg.optimizer.lbfgs_iterations:
            if self._budget_reached(max_iterations):
                return False
            iteration = self.global_iteration
            first: dict = {}

            def closure():
                self.lbfgs.zero_grad(set_to_none=True)
                terms, weights, total, interior = self._evaluate(points, iteration)
                if not first:  # losses at the start of this iteration
                    first.update(terms={k: v.detach() for k, v in terms.items()}, weights=weights,
                                 total=total.detach(), interior=interior)  # fmt: skip
                if bool(torch.isfinite(total)):
                    total.backward()
                return total

            self.lbfgs.step(closure)
            self._check_finite(first["total"])
            if self.phase_iter % self.cfg.training.lbfgs_log_every == 0:
                self._log("lbfgs", first["terms"], first["weights"], first["total"], first["interior"],
                          lr=self.lbfgs.param_groups[0]["lr"])  # fmt: skip
            self.phase_iter += 1
            self._maybe_checkpoint()
        self.phase = "done"
        return True

    def _final_evaluation(self) -> dict:
        terms, weights, total, interior = self._evaluate(self._points(self._lbfgs_index()), self.global_iteration)
        self._check_finite(total)
        self._log("final", terms, weights, total, interior, lr=0.0)
        total_f = float(total.detach())
        groups = group_values(terms, self.groups)
        return {
            "total": total_f,
            "terms": {t: float(terms[t].detach()) for t in self.terms},
            "weights": weights,
            "dominance": {g: weights[g] * groups[g] / total_f if total_f > 0 else float("nan") for g in self.groups},
        }

    # Entry point ------------------------------------------------------------------
    def run(self, max_iterations: int | None = None) -> dict:
        """Train until done (or until the global iteration reaches ``max_iterations``).

        Returns the run summary; also written to summary.json with an output directory.
        """
        cfg = self.cfg
        self._t0, self._elapsed_base = time.perf_counter(), self.elapsed
        losses_path = None if self.out_dir is None else self.out_dir / "losses.csv"
        self.logger = LossLogger(losses_path, self._columns(),
                                 keep_before_iteration=self.global_iteration if self._resumed else None)  # fmt: skip

        if self.precision is None:
            X = sample_interior(self.domain, cfg.diagnostics.precision_check_points, seed=cfg.experiment.seed,
                                index=DIAGNOSTIC_STREAM, dtype=self.dtype, device=self.device)  # fmt: skip
            self.precision = precision_check(self.model, X, self.scales, self.stage)

        status, message, final, diagnostics = "completed", "", None, None
        try:
            if self.phase == "adam" and not self._run_adam(max_iterations):
                status = "interrupted"
            if status == "completed" and self.phase == "lbfgs" and not self._run_lbfgs(max_iterations):
                status = "interrupted"
            if status == "completed":
                self.phase = "done"
                final = self._final_evaluation()
                diagnostics = run_diagnostics(self.model, cfg, self.scales, self.domain, self.loading)
        except NonFiniteLossError as exc:
            status, message = "failed", str(exc)

        if status == "completed":
            self._save("final.pt")
        if status != "failed":  # never overwrite the last good checkpoint with a broken state
            self._save("latest.pt")
        self.elapsed = self._wall()

        summary = self._summary(status, message, final, diagnostics)
        if self.out_dir is not None:
            write_json(self.out_dir / "summary.json", summary)
        return summary

    def _summary(self, status, message, final, diagnostics) -> dict:
        cfg, g = self.cfg, self.global_iteration
        n_adam = cfg.optimizer.adam_iterations
        summary = {
            "status": status,
            "message": message,
            "experiment_id": cfg.experiment.id,
            "stage": self.stage,
            "constraints_mode": self.mode,
            "dtype": cfg.precision.dtype,
            "device": str(self.device),
            "device_name": describe_device(self.device),
            "config_hash": cfg.config_hash,
            "iterations": {"adam": min(g, n_adam), "lbfgs": max(g - n_adam, 0), "total": g},
            "training_time_s": round(self.elapsed, 3),
            "resume_count": self.resume_count,
            "precision_check": self.precision,
            "final": final,
            "diagnostics": diagnostics,
        }
        if self.stage == 2:
            summary["T_hat"] = cfg.time.T_hat
            summary["T_ms"] = self.scales.time_ms(cfg.time.T_hat)
        return summary
