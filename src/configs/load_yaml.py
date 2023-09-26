import yaml

def load_yaml(dir):
    return yaml.load(
        open(dir, 'r'),
        Loader=yaml.FullLoader
    )