__all__ = [
    "AudioVectorizer"
]

from abc import abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np

from radient.tasks.vectorizers._base import Vectorizer
from radient.utils import fully_qualified_name
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

librosa = LazyImport("librosa")


class AudioVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    def _preprocess(self, audio: Any, **kwargs) -> np.ndarray:
        if isinstance(audio, tuple) and isinstance(audio[0], np.ndarray):
            waveform, source_rate = audio
        elif isinstance(audio, str):
            waveform, source_rate = librosa.load(audio, sr=None, mono=False)
            if len(waveform.shape) == 1:
                waveform = np.expand_dims(waveform, 0)

        if source_rate != self.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=source_rate,
                target_sr=self.sample_rate
            )

        return waveform

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Returns the sample rate required by this vectorizer.
        """
        pass
    