# Runs

Committed experiment outputs, one folder per run:

```
runs/EXPnnn/<UTC-timestamp>_<stage>_seed<k>/
```

Folders here are written by `scripts/run.py` (normally on Colab) and pushed by
`scripts/push_results.py`. Do not edit them by hand.

Contents of each run folder: see [`docs/run_folder_format.md`](../docs/run_folder_format.md).
