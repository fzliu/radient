from radient.tasks.transforms._base import Transform
from radient.tasks.transforms.video_demux import VideoDemuxTransform


def transform(method: str = "video_demux", **kwargs) -> Transform:

    if method == "video_demux":
        return VideoDemuxTransform(**kwargs)
    else:
        raise ValueError(f"unknown transform method: {method}")
