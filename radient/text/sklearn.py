__all__ = [
    "CountTextVectorizer"
]

from typing import Dict, List, Optional
import warnings

from radient.base import Vector
from radient.util import LazyImport
from radient.text.base import TextVectorizer

CountVectorizer = LazyImport("sklearn.feature_extraction.text", attribute="CountVectorizer", package="scikit-learn")
TfidfVectorizer = LazyImport("sklearn.feature_extraction.text", attribute="TfidfVectorizer", package="scikit-learn")
HashingVectorizer = LazyImport("sklearn.feature_extraction.text", attribute="HashingVectorizer", package="scikit-learn")


class SklearnTextVectorizer(TextVectorizer):
    """Text vectorization with `sentence-transformers`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._model = CountVectorizer(**kwargs)

    def vectorize(self, texts: List[str]) -> List[Vector]:
        #TODO(fzliu): token length check
        texts = TextVectorizer.standardize_inputs(texts)
        vectors = self._model.fit_transform(texts)
        return [v.view(Vector) for v in vectors]
