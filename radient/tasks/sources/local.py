from collections.abc import Iterator
from pathlib import Path
from typing import Dict

from radient.tasks.sources._base import Source


def _path_walk(path):
    p = Path(path)
    if p.is_file():
        yield path
    elif p.is_dir():
        for sub in p.iterdir():
            yield from _path_walk(sub)


class LocalSource(Source):
    """Reads filenames from a local directory. This source is mostly useful for
    backfills from local disk and or for testing purposes.
    """

    def __init__(self, path: str, **kwargs):
        super().__init__()
        self._paths_iter = _path_walk(path)

    def read(self) -> Dict[str, str]:
        try:
            return {"data": self._paths_iter.__next__()}
        except StopIteration:
            return None
