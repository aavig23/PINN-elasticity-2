"""Loss weighting strategies (spec §5.5, decision B5).

A strategy is called as strategy(terms, iteration) -> {group: weight} for the
active groups. Only fixed weights are implemented for the baselines; adaptive
strategies plug in behind the same interface when an experiment needs one.
"""


class FixedWeights:
    def __init__(self, weights: dict[str, float], groups):
        self.weights = {g: float(weights[g]) for g in groups}

    def __call__(self, terms, iteration: int) -> dict[str, float]:
        return dict(self.weights)


def build_weighting(loss_cfg, groups):
    if loss_cfg.weighting == "fixed":
        return FixedWeights(loss_cfg.weights, groups)
    raise ValueError(f"unknown weighting strategy {loss_cfg.weighting!r}")
