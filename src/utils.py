from ruamel.yaml import YAML
import pathlib

def load_dvc_params():
    root = pathlib.Path(__file__).parent.parent.resolve()
    print(root)
    with open(f"{root}/params.yaml", "r") as f:
       yaml = YAML()
       params = yaml.load(f)
    return params