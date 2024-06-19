__all__ = [
    "VoyageTextVectorizer"
]

from typing import List

from radient.tasks.vectorizers.text._base import TextVectorizer
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

voyageai = LazyImport("voyageai")


class VoyageTextVectorizer(TextVectorizer):
    """Text vectorization with Voyage AI (https://www.voyageai.com).
    """

    def __init__(self, model_name: str = "voyage-2", **kwargs):
        super().__init__()
        self._model_name = model_name
        self._client = voyageai.Client()

    def _vectorize(self, text: str, **kwargs) -> Vector:
        res = self._client.embed(text, model=self._model_name)
        return np.array(res.embeddings).view(Vector)

    @property
    def model_name(self):
        return self._model_name
