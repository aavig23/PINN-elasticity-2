"""Machine-readable training logs: losses.csv and summary.json (spec §7.4)."""

import csv
import json
from pathlib import Path


class LossLogger:
    """Append rows to losses.csv (if a path is given) and keep them in memory.

    On resume, pass ``keep_before_iteration``: rows logged at or after that
    iteration by the interrupted session are dropped, because the resumed run
    logs them again.
    """

    def __init__(self, path: Path | None, columns: list[str], keep_before_iteration: int | None = None):
        self.path = None if path is None else Path(path)
        self.columns = list(columns)
        self.rows: list[dict] = []
        if self.path is None:
            return
        kept = []
        if keep_before_iteration is not None and self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as fh:
                kept = [r for r in csv.DictReader(fh) if int(r["iteration"]) < keep_before_iteration]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(kept)

    def log(self, row: dict) -> None:
        self.rows.append(row)
        if self.path is None:
            return
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.columns).writerow(row)


def read_losses(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n", encoding="utf-8")
