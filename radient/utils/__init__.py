import urllib.request
from pathlib import Path

from typing import Any, Optional


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