"""The committed Colab notebook is clean: no outputs, no secrets, clones this repo."""

import json
import re

from conftest import ROOT

NOTEBOOK = ROOT / "notebooks" / "colab_runner.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_nbformat_4():
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert all(isinstance(c["source"], list) for c in nb["cells"])


def test_notebook_has_no_saved_outputs():
    for cell in _code_cells(_notebook()):
        assert cell["outputs"] == [] and cell["execution_count"] is None


def test_notebook_contains_no_tokens():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|x-access-token", text)
    assert "userdata" not in text  # no secret access until the push step exists


def test_notebook_clones_this_repo_and_keeps_colab_torch():
    source = "".join(line for c in _code_cells(_notebook()) for line in c["source"])
    assert "https://github.com/aavig23/PINN-elasticity-2.git" in source
    assert "pip install --quiet --no-deps -e ." in source
    assert not re.search(r"pip install[^\n]*\btorch\b", source)
    assert "scripts/check_environment.py --install-missing --require-gpu" in source
    assert "scripts/smoke_test.py" in source
