__all__ = [
    "VideoDemuxTransform"
]

from typing import Optional

from radient.tasks.transforms.video_demux._base import VideoDemuxTransform
from radient.tasks.transforms.video_demux.default import DefaultVideoDemuxTransform
from radient.tasks.transforms.video_demux.ffmpeg import FFmpegVideoDemuxTransform


def video_demux_transform(method: str = "default", **kwargs) -> VideoDemuxTransform:
    """Creates a video demultiplexer specified by `method`.
    """

    # Return a reasonable default vectorizer in the event that the user does
    # not specify one.
    if method in ("default", None):
        return DefaultVideoDemuxTransform(**kwargs)
    elif method in ("ffmpeg",):
        return FFmpegVideoDemuxTransform(**kwargs)
    else:
        raise NotImplementedError
