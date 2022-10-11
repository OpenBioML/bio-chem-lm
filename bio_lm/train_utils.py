import yaml


def load_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config