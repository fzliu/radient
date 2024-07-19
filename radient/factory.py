from typing import Dict, Optional, Tuple, Type 

from radient.orchestrate.runners import *
from radient.tasks.sinks import *
from radient.tasks.sources import *
from radient.tasks.transforms import *
from radient.tasks.vectorizers import *



def make_operator(
    optype: str,
    method: str,
    modality: Optional[str] = None,
    runner: Optional[Type] = None,
    task_params: Optional[Dict] = None
) -> Runner:

    runner = runner or LocalRunner
    task_params = task_params or {}

    # Create a data sink.
    if optype == "sink":
        if method == "milvus":
            return runner(MilvusSink, task_params=task_params)
        else:
            raise ValueError(f"unknown data store: {method}")

    # Create a data source.
    elif optype == "source":
        if method == "local":
            return runner(LocalSource, task_params=task_params)
        elif method == "youtube":
            return runner(YoutubeSource, task_params=task_params)
        else:
            raise ValueError(f"unknown data source: {method}")

    # Create a unstructured data to vector transformation.
    elif optype == "transform":
        if method == "video-demux":
            return runner(VideoDemuxTransform, task_params=task_params)
        else:
            raise ValueError(f"unknown transform method: {method}")

    # Create an unstructured data vectorizer.
    elif optype == "vectorizer":
        task_params["method"] = method
        if modality == "audio":
            return runner(audio_vectorizer, task_params=task_params)
        elif modality == "graph":
            return runner(graph_vectorizer, task_params=task_params)
        elif modality == "image":
            return runner(image_vectorizer, task_params=task_params)
        elif modality == "molecule":
            return runner(molecule_vectorizer, task_params=task_params)
        elif modality == "text":
            return runner(text_vectorizer, task_params=task_params)
        elif modality == "multimodal":
            return runner(multimodal_vectorizer, task_params=task_params)
        else:
            raise NotImplementedError

