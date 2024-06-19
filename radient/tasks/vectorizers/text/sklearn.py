__all__ = [
    "CountTextVectorizer"
]

from typing import Dict, List, Optional
import warnings

from radient.tasks.vectorizers.text._base import TextVectorizer
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

CountVectorizer = LazyImport("sklearn.feature_extraction.text", attribute="CountVectorizer", package_name="scikit-learn")
TfidfVectorizer = LazyImport("sklearn.feature_extraction.text", attribute="TfidfVectorizer", package_name="scikit-learn")
HashingVectorizer = LazyImport("sklearn.feature_extraction.text", attribute="HashingVectorizer", package_name="scikit-learn")


class SklearnTextVectorizer(TextVectorizer):
    """Text vectorization with `sentence-transformers`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._model = CountVectorizer(**kwargs)

    def _vectorize(self, texts: str, **kwargs) -> Vector:
        vectors = self._model.transform(texts)
        # TODO(fzliu): sparse vector type
        return vectors
        #return [v.view(Vector) for v in vectors]
