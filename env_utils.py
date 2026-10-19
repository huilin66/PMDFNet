import os
from pathlib import Path


def load_env_file(path=None):
    env_path = Path(path) if path else Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        return {}

    try:
        from dotenv import dotenv_values, load_dotenv

        load_dotenv(env_path, override=False)
        values = {
            key: value
            for key, value in dotenv_values(env_path).items()
            if value is not None
        }
        return values
    except ImportError:
        pass

    values = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            values[key] = value
            os.environ.setdefault(key, value)
    return values


_ENV = load_env_file()


def get_env_value(*names, default=""):
    for name in names:
        value = os.environ.get(name) or _ENV.get(name)
        if value:
            return value
    return default


def get_data_root(default=""):
    return get_env_value("MMLSV2_DATA_ROOT", "MARS_DATA_ROOT", "DATA_ROOT", default=default)
