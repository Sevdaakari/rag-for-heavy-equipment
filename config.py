import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config(path=_CONFIG_PATH):
    with open(path, "r") as f:
        return json.load(f)


CONFIG = load_config()