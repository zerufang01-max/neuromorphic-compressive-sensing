"""Project paths, explicit checkpoint manifests, and JSON persistence."""
import importlib.util
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / 'outputs'
PAPER_CONFIG = PROJECT_DIR / 'configs' / 'paper.json'


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def selected_models(output):
    output = Path(output).expanduser().resolve()
    rows = read_json(output / 'configs' / 'selected_configs.json')
    selected = {}
    for row in rows:
        if row['method'] == 'slista' and row['time_steps'] != 1:
            continue
        key = (row['condition']['id'], row['id'])
        if key in selected:
            raise ValueError(f'Duplicate model in manifest: {key}')
        path = output / row['checkpoint']
        if not path.is_file():
            raise FileNotFoundError(path)
        selected[key] = (row, path)
    return selected


def resolve_module(root, name):
    path = Path(root) / (name + '.py')
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path
