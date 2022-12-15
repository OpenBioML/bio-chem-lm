from functools import partial

from torchmetrics import (AUROC, MeanMetric, MeanSquaredError, PearsonCorrCoef,
                          Precision)

name2metric = {
    "rmse": partial(MeanSquaredError, squared=False),
    "pearsonr": PearsonCorrCoef,
    "precision": partial(Precision, task="binary"),
    "roc": partial(AUROC, task="binary"),
}

PROBLEM2METRICS = {
    "regression": ["rmse", "pearsonr"],
    "classification": ["roc", "precision"],
}


class MetricDict:
    def __init__(self, names, device, name2metric=None) -> None:
        if names:
            self.metrics = {name: MeanMetric().to(device) for name in names}

        if name2metric:
            for metric_name, metric in name2metric.items():
                self.metrics[metric_name] = metric().to(device)

    def update(self, values):
        for name in self.metrics:
            self.metrics[name].update(**values[name])

    def compute(self):
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()


def format_metrics(metrics, split, prefix=""):
    log = f"[{split}]" + prefix
    log += " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

    return log
