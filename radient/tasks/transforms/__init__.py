from radient.tasks.transforms._base import Transform
from radient.tasks.transforms.video_demux import VideoDemuxTransform


def transform(method: str = "video-demux", **kwargs) -> Transform:

    if method == "video-demux":
        return VideoDemuxTransform(**kwargs)
    else:
        raise ValueError(f"unknown transform method: {method}")
