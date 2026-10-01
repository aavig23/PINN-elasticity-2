"""Report and check the Python environment for this project (local or Colab).

    python scripts/check_environment.py                    # report; fail on hard problems
    python scripts/check_environment.py --require-gpu      # also fail without a CUDA GPU
    python scripts/check_environment.py --install-missing  # Colab: install absent packages first

Dependency policy (spec §5.2): torch is never installed or upgraded here; on
Colab the preinstalled, CUDA-matched build is used. Other dependencies are
installed only if completely missing. A package older than the project's
lower bound is reported as a warning, not upgraded, because upgrading
Colab's preinstalled packages can break others that depend on them.

Exit code 0 = all checks passed, 1 = at least one failure.
"""

import argparse
import importlib
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import name and distribution name for each requirement in pyproject.toml.
REQUIREMENTS = [
    # (distribution, import name, lower bound, role)
    ("torch", "torch", "2.6", "runtime"),
    ("numpy", "numpy", "2.1", "runtime"),
    ("matplotlib", "matplotlib", "3.9", "runtime"),
    ("pyyaml", "yaml", "6.0.2", "runtime"),
    ("pytest", "pytest", "8", "tests"),
]
NEVER_INSTALL = {"torch"}


def _version_tuple(text: str) -> tuple:
    parts = []
    for piece in text.split("+")[0].split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def _installed_version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--require-gpu", action="store_true", help="fail if no CUDA GPU is available")
    parser.add_argument("--install-missing", action="store_true", help="pip-install absent non-torch packages")
    args = parser.parse_args()

    failures, warnings = [], []
    # Colab sets COLAB_RELEASE_TAG for the runtime; it is inherited by `!python` subprocesses.
    on_colab = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules

    print("== Platform ==")
    print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"  platform    {platform.platform()}")
    print(f"  colab       {'yes' if on_colab else 'no'}")

    if args.install_missing:
        missing = [d for d, _, _, _ in REQUIREMENTS if _installed_version(d) is None and d not in NEVER_INSTALL]
        if missing:
            specs = [f"{d}>={lb}" for d, _, lb, _ in REQUIREMENTS if d in missing]
            print(f"\n== Installing missing packages: {' '.join(specs)} ==")
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *specs], check=True)
        else:
            print("\n== Installing missing packages: none missing ==")

    print("\n== Dependencies (lower bounds from pyproject.toml) ==")
    for dist, module, lower, role in REQUIREMENTS:
        version = _installed_version(dist)
        if version is None:
            failures.append(f"{dist} is not installed")
            print(f"  {dist:<11} MISSING   (need >={lower}, {role})")
            continue
        ok = _version_tuple(version) >= _version_tuple(lower)
        if not ok:
            warnings.append(f"{dist} {version} is older than the project's lower bound {lower}")
        print(f"  {dist:<11} {version:<14} {'ok' if ok else 'BELOW BOUND (warning)'}  (need >={lower}, {role})")

    print("\n== PyTorch / GPU ==")
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  torch       {torch.__version__}")
        print(f"  cuda build  {torch.version.cuda or 'none (CPU-only build)'}")
        print(f"  cudnn       {torch.backends.cudnn.version() if cuda else 'n/a'}")
        print(f"  gpu         {torch.cuda.get_device_name(0) if cuda else 'none'}")
        if cuda:
            x = torch.randn(256, 256, device="cuda")
            print(f"  gpu matmul  ok (checksum {float((x @ x).abs().mean()):.3f})")
        elif args.require_gpu:
            failures.append("no CUDA GPU available (Colab: Runtime -> Change runtime type -> GPU)")
    except Exception as exc:  # noqa: BLE001 - report any import/runtime problem
        failures.append(f"torch unusable: {exc}")

    print("\n== Project package ==")
    try:
        pkg = importlib.import_module("pinn_elasticity")
        location = Path(pkg.__file__).resolve()
        from_repo = ROOT / "src" in location.parents
        print(f"  version     {pkg.__version__}")
        print(f"  imported    {location}")
        if not from_repo:
            failures.append(f"pinn_elasticity is imported from {location}, not from this repository's src/")
        for sub in ("config", "physics.equations", "models.constraints", "training.trainer", "sampling"):
            importlib.import_module(f"pinn_elasticity.{sub}")
        print("  submodules  ok")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"pinn_elasticity does not import: {exc}")

    print("\n== Repository ==")
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    print(f"  commit      {commit or 'unknown (not a git checkout?)'}")
    print(f"  branch      {_git('rev-parse', '--abbrev-ref', 'HEAD') or 'unknown'}")
    print(f"  clean       {'yes' if not dirty else 'NO - local changes present'}")
    if dirty:
        warnings.append("working tree has local changes; a run would not match the recorded commit")

    print("\n== Result ==")
    for w in warnings:
        print(f"  WARNING: {w}")
    for f in failures:
        print(f"  FAIL:    {f}")
    print("  ALL CHECKS PASSED" if not failures else f"  {len(failures)} CHECK(S) FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
