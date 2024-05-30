from radient.transforms.video_demux import VideoDemuxTransform


def make_transform(method: str = "video_demux", **kwargs):

    if method == "video_demux":
        return VideoDemuxTransform(**kwargs)
    else:
        raise ValueError(f"unknown transform method: {method}")
