__all__ = [
    "TimmImageVectorizer"
]

from typing import Any, List

from radient.base import Vector
from radient.util import LazyImport
from radient.image.base import AudioVectorizer

torchaudio = LazyImport("torchaudio")
torch = LazyImport("torch")


class TorchaudioAudioVectorizer(AudioVectorizer):
    """Computes audio embeddings using `torchaudio`.
    """

    def __init__(self, model_name: str = "wave2vec2", **kwargs):
        super().__init__()
        self._model_name = model_name

    def vectorize(self, images: List[Any]) -> List[Vector]:
        pass

    def accelerate(self, **kwargs):
        pass



