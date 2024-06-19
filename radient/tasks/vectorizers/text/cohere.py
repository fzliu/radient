__all__ = [
    "CohereTextVectorizer"
]

import os
from typing import List, Optional

from radient.utils.lazy_import import LazyImport
from radient.vector import Vector
from radient.tasks.vectorizers.text._base import TextVectorizer

cohere = LazyImport("cohere")


class CohereTextVectorizer(TextVectorizer):
    """Text vectorization with Cohere (https://www.cohere.com).
    """

    def __init__(self, model_name: str = "embed-english-v3.0", **kwargs):
        super().__init__()
        self._model_name = model_name
        if "COHERE_API_KEY" in os.environ:
            api_key = os.envirion["COHERE_API_KEY"]
        elif api_key in kwargs:
            api_key = kwargs["api_key"]
        else:
            raise ValueError("API key not found")
        self._client = cohere.Client(api_key)

    def _vectorize(self, text: str, **kwargs) -> Vector:
        vector = self._client.embed([text], model=self._model_name)
        return vector.view(Vector)

    @property
    def model_name(self):
        return self._model_name
