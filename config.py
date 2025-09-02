import os, yaml
import shutil

# Check if config.yml exists, if not copy from config-example.yml
if not os.path.exists(os.path.join(os.path.dirname(__file__), "config.yml")):
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "config-example.yml"),
        os.path.join(os.path.dirname(__file__), "config.yml"),
    )

with open(os.path.join(os.path.dirname(__file__), "config.yml"), "r") as f:
    cfg = yaml.safe_load(f)
