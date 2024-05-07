__all__ = [
    "AudioVectorizer"
]

from abc import abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np

from radient.util.lazy_import import fully_qualified_name, LazyImport
from radient.vectorizers.base import Vector, Vectorizer

torchaudio = LazyImport("torchaudio")


class AudioVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    def _preprocess(cls, audio: Any) -> str:
        if isinstance(audio, tuple) and isinstance(audio[0], np.ndarray):
            return audio
        elif isinstance(audio, str):
            return torchaudio.load(audio)