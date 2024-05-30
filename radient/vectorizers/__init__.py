from collections.abc import Callable

from radient.vectorizers.base import Vectorizer
from radient.vectorizers.audio import audio_vectorizer
from radient.vectorizers.graph import graph_vectorizer
from radient.vectorizers.image import image_vectorizer
from radient.vectorizers.molecule import molecule_vectorizer
from radient.vectorizers.text import text_vectorizer
from radient.vectorizers.multimodal import multimodal_vectorizer


def make_vectorizer(
    modality: str = "text",
    method: str = "sbert",
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
