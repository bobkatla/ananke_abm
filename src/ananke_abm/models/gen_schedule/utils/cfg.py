
import json
import yaml
import os

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml",".yml")):
            return yaml.safe_load(f)
        return json.load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
