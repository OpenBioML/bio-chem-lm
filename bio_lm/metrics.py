from torchmetrics import MeanMetric


class MetricDict:
    def __init__(self, names, device) -> None:
        self.metrics = {name: MeanMetric().to(device) for name in names}

    def update(self, values):
        for name in self.metrics:
            self.metrics[name].update(values[name])

    def compute(self):
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()


def format_metrics(metrics, split, prefix=""):
    log = f"[{split}]" + prefix
    log += " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

    return log
