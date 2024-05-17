import importlib
import importlib.util
from pathlib import Path
import subprocess
from types import ModuleType
from typing import Any, Dict, List, Optional
import urllib.request
import warnings

import pip


def download_cache_file(url: str, filename: Optional[str] = None) -> None:
    """Download a file from a URL and save it to a local file.
    """
    if not filename:
        filename = url.split("/")[-1].split("?")[0]
    path = Path.home() / ".cache" / "radient" / filename
    path.parents[0].mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def fully_qualified_name(instance: Any) -> str:
    return f"{instance.__class__.__module__}.{instance.__class__.__qualname__}"


def prompt_install(package: str, version: str) -> bool:
    """Checks whether the user wants to install a module before proceeding.
    """
    if version:
        package = f"{package}>={version}"
    # Ignore "no" responses in `prompt_install` so the user can optionally
    # install it themselves when prompted.
    if input(f"Vectorizer requires {package}. Install? [Y/n]\n") == "Y":
        if subprocess.check_call(["pip", "install", "-q", package]) == 0:
            return True
        else:
            warnings.warn(f"Could not install required package {package}")
    return False


class LazyImport(ModuleType):
    """Lazily import a module to avoid unnecessary dependencies. If a required
    dependency does not exist, it will prompt the user for it.

    Adapted from tensorflow/python/util/lazy_loader.py.
    """

    def __init__(
        self,
        name: str,
        attribute: Optional[str] = None,
        package_name: Optional[str] = None,
        min_version: Optional[str] = None
    ):
        super().__init__(name)
        self._attribute = attribute
        self._top_name = name.split(".")[0]
        self._package_name = package_name if package_name else self._top_name
        self._min_version = min_version
        self._module = None

    def _load(self) -> ModuleType:
        if not self._module:
            if not importlib.util.find_spec(self._top_name):
                prompt_install(self._package_name, self._min_version)
        self._module = importlib.import_module(self.__name__)
        if self._min_version and self._module.__version__ < self._min_version:
            prompt_install(self._package_name, self._min_version)
            self._module = importlib.import_module(self.__name__)
        if self._attribute:
            return getattr(self._module, self._attribute)
        return self._module

    def __call__(self, *args, **kwargs) -> Any:
        return self._load()(*args, **kwargs)

    def __getattr__(self, attribute: Any) -> Any:
        return getattr(self._load(), attribute)

    def __dir__(self) -> List:
        return dir(self._load())

