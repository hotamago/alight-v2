import os, yaml

with open(os.path.join(os.path.dirname(__file__), 'config.yml'), 'r') as f:
    cfg = yaml.safe_load(f)
