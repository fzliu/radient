__all__ = [
    "AudioVectorizer"
]

from abc import abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np

from radient.utils import fully_qualified_name, LazyImport
from radient.vectorizers.base import Vector, Vectorizer

torchaudio = LazyImport("torchaudio")


class AudioVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    def _preprocess(self, audio: Any) -> np.ndarray:
        if isinstance(audio, tuple) and isinstance(audio[0], np.ndarray):
            pass
        elif isinstance(audio, str):
            audio = torchaudio.load(audio)
        if audio[1] != self.sample_rate:
            wave = torchaudio.functional.resample(*audio, self.sample_rate)
        else:
            wave = audio[0]
        return wave

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Returns the sample rate required by this vectorizer.
        """
        raise NotImplementedError
    