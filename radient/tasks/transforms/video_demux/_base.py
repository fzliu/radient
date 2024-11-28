from abc import abstractmethod
from pathlib import Path
import uuid

from radient.tasks.transforms._base import Transform


class VideoDemuxTransform(Transform):

    def __init__(self,
        interval: float = 2.0,
        output_directory: str = "~/.radient/data/video_demux",
        **kwargs
    ):
        super().__init__()
        self._interval = interval
        output_directory = Path(output_directory).expanduser()
        self._output_directory = output_directory

    @abstractmethod
    def transform(self, data: str) -> dict[str, list[str]]:
        """Extracts frames and audio snippets from a video file.
        """
        pass

    def _make_output_dir(self):
        # The full output directory comes from a combination of the user's
        # specification plus a unique identifier for the current run.
        # TODO(fzliu): switch to an incremental identifier e.g. UUIDv7
        output_path = Path(self._output_directory) / str(uuid.uuid4())
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
