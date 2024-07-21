from collections.abc import Iterator
from pathlib import Path
from typing import Dict
import uuid

from radient.tasks.sources._base import Source
from radient.utils.lazy_import import LazyImport

yt_dlp = LazyImport("yt_dlp")


class YoutubeSource(Source):
    """Downloads videos from Youtube to a local directory. The `url` argument
    can be a single video or a playlist.
    """

    def __init__(self,
            url: str,
            output_directory: str = "~/.radient/data/youtube",
            **kwargs
        ):
        super().__init__()
        self._filename = None

        # Create a new output directory for downloading and storing the videos.
        output_directory = Path(output_directory).expanduser()
        output_directory = output_directory / str(uuid.uuid4())
        output_directory.mkdir(parents=True, exist_ok=True)
        kwargs["outtmpl"] = str(output_directory / "%(id)s.%(ext)s")

        # The input URL may be a single video or a playlist of videos. Here, we
        # extract a list of all video URLs for use in the `read` function.
        with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        if "entries" in info:
            self._video_urls = [entry["webpage_url"] for entry in info["entries"]]
        else:
            self._video_urls = [info["webpage_url"]]
        self._url_idx = 0

        # Add a hook to dynamically determine what the output filename is for
        # each video.
        def hook(d):
            if d["status"] == "finished":
                self._filename = d["filename"]
        kwargs["progress_hooks"] = [hook]
        self._youtube_dl = yt_dlp.YoutubeDL(kwargs)

    def read(self) -> Dict[str, str]:

        if self._url_idx == len(self._video_urls):
            return None
        self._youtube_dl.download(self._video_urls[self._url_idx])
        self._url_idx += 1
        return {"data": self._filename}




