from typing import Optional, Type 

from radient.orchestrate.runners import *
from radient.tasks.sinks import *
from radient.tasks.sources import *
from radient.tasks.transforms import *
from radient.tasks.vectorizers import *


def make_operator(
    task_name: str,
    task_type: str,
    runner: Optional[Type] = None,
    task_params: Optional[dict] = None
) -> Runner:

    runner = runner or LocalRunner
    task_params = task_params or {}

    # Create a data sink.
    if task_name == "sink":
        if task_type == "milvus":
            return runner(MilvusSink, task_params=task_params)
        elif task_type == "mongodb":
            return runner(MongoDBSink, task_params=task_params)
        else:
            raise ValueError(f"unknown data store: {task_type}")

    # Create a data source.
    elif task_name == "source":
        if task_type == "local":
            return runner(LocalSource, task_params=task_params)
        elif task_type == "youtube":
            return runner(YoutubeSource, task_params=task_params)
        elif task_type == "ingest":
            return runner(IngestSource, task_params=task_params)
        else:
            raise ValueError(f"unknown data source: {task_type}")

    # Create a data-to-data transformation.
    elif task_name == "transform":
        if task_type == "video-demux":
            return runner(video_demux_transform, task_params=task_params)
        elif task_type == "speech-to-text":
            return runner(speech_to_text_transform, task_params=task_params)
        else:
            raise ValueError(f"unknown transform method: {task_type}")

    # Create an data-to-vector transformation.
    elif task_name == "vectorizer":
        if task_type == "audio":
            return runner(audio_vectorizer, task_params=task_params)
        elif task_type == "graph":
            return runner(graph_vectorizer, task_params=task_params)
        elif task_type == "image":
            return runner(image_vectorizer, task_params=task_params)
        elif task_type == "molecule":
            return runner(molecule_vectorizer, task_params=task_params)
        elif task_type == "text":
            return runner(text_vectorizer, task_params=task_params)
        elif task_type == "multimodal":
            return runner(multimodal_vectorizer, task_params=task_params)
        else:
            raise NotImplementedError

