import yaml
import warnings

from pathlib import Path


class Configuration(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._project_dir = Path(__file__).resolve().parent.parent.parent

        self.save()
            #print(f"Loaded {self}")

    def __setitem__(self, key, value):
        """ saves the string representation or list of the input into the config and saves the config to the file"""
        if isinstance(value, list):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, str(value))

        with open(Path(self._project_dir, "res", "file_config.yml"), "w") as file:
            yaml.dump(self, file)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        elif not default is None:
            return default
        else:
            raise ValueError(f"{key} not found in Configuration and no default specified")

    def save(self):
        with open(Path(self._project_dir, "res", "file_config.yml"), "r") as file:
            self.update(yaml.safe_load(file))

