from pathlib import Path
import shutil
from typing import Dict

from radient.tasks.sources._base import Source


class IngestSource(Source):

    def __init__(self, path: str, **kwargs):
        super().__init__()
        source = Path(path).expanduser()
        destination = Path("~") / ".radient" / "data" / "ingest"
        destination = destination.expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination / source.name)

    def read(self) -> Dict[str, str]:
        return None
