__all__ = [
    "TextVectorizer",
    "SBERTTextVectorizer",
    "SklearnTextVectorizer",
    "text_vectorizer"
]

from typing import Optional

from radient.vectorizers.text.base import TextVectorizer
from radient.vectorizers.text.sbert import SBERTTextVectorizer
from radient.vectorizers.text.sklearn import SklearnTextVectorizer
from radient.vectorizers.text.voyage import VoyageTextVectorizer


def text_vectorizer(method: Optional[str] = None, **kwargs) -> TextVectorizer:
    """Creates a text vectorizer specified by `method`.
    """

    # Return a reasonable default vectorizer in the event that the user does
    # not specify one.
    if method in (None, "sbert", "sentence-transformers"):
        return SBERTTextVectorizer(**kwargs)
    elif method in ("sklearn", "scikit-learn"):
        return SklearnTextVectorizer(**kwargs)
    elif method in ("voyage", "voyageai"):
        return VoyageTextVectorizer(**kwargs)
    else:
        raise NotImplementedError
