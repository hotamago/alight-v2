import os, yaml
import shutil

# Check if config.yml exists, if not copy from config-example.yml
if not os.path.exists(os.path.join(os.path.dirname(__file__), "config.yml")):
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "config-example.yml"),
        os.path.join(os.path.dirname(__file__), "config.yml"),
    )

# Load both config.yml and config-example.yml
with open(os.path.join(os.path.dirname(__file__), "config.yml"), "r") as f:
    cfg = yaml.safe_load(f)
with open(os.path.join(os.path.dirname(__file__), "config-example.yml"), "r") as f:
    cfg_example = yaml.safe_load(f)


# if exit then check diffirent deep key (dict in dict, list, .get) on config-example.yml and config.yml if not exit on config.yml then add to config.yml
def smart_deep_copy(dict1, dict2):
    isChange = False
    for key, value in dict1.items():
        if key not in dict2:
            dict2[key] = value
            isChange = True
        else:
            if isinstance(value, dict):
                isChange = smart_deep_copy(value, dict2[key])
            else:
                if dict2[key] != value:
                    dict2[key] = value
                    isChange = True

    return isChange


isChange = smart_deep_copy(cfg_example, cfg)
if isChange:
    with open(os.path.join(os.path.dirname(__file__), "config.yml"), "w") as f:
        yaml.dump(cfg, f)
