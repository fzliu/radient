from collections.abc import Callable

from radient.tasks.vectorizers._base import Vectorizer
from radient.tasks.vectorizers.audio import audio_vectorizer
from radient.tasks.vectorizers.graph import graph_vectorizer
from radient.tasks.vectorizers.image import image_vectorizer
from radient.tasks.vectorizers.molecule import molecule_vectorizer
from radient.tasks.vectorizers.text import text_vectorizer
from radient.tasks.vectorizers.multimodal import multimodal_vectorizer


def vectorizer(
    modality: str = "multimodal",
    method: str = "imagebind",
    **kwargs
) -> Vectorizer:

    if modality == "audio":
        return audio_vectorizer(method=method, **kwargs)
    elif modality == "graph":
        return graph_vectorizer(method=method, **kwargs)
    elif modality == "image":
        return image_vectorizer(method=method, **kwargs)
    elif modality == "molecule":
        return molecule_vectorizer(method=method, **kwargs)
    elif modality == "text":
        return text_vectorizer(method=method, **kwargs)
    elif modality == "multimodal":
        return multimodal_vectorizer(method=method, **kwargs)
    else:
        raise NotImplementedError
