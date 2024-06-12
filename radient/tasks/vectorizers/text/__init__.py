__all__ = [
    "TextVectorizer",
    "CohereTextVectorizer"
    "ImageBindTextVectorizer"
    "SBERTTextVectorizer",
    "SklearnTextVectorizer",
    "VoyageTextVectorizer"
    "text_vectorizer"
]

from typing import Optional

from radient.tasks.vectorizers.text._base import TextVectorizer
from radient.tasks.vectorizers.text.cohere import CohereTextVectorizer
from radient.tasks.vectorizers.text.imagebind import ImageBindTextVectorizer
from radient.tasks.vectorizers.text.sbert import SBERTTextVectorizer
from radient.tasks.vectorizers.text.sklearn import SklearnTextVectorizer
from radient.tasks.vectorizers.text.voyage import VoyageTextVectorizer


def text_vectorizer(method: str = "sbert", **kwargs) -> TextVectorizer:
    """Creates a text vectorizer specified by `method`.
    """

    # Return a reasonable default vectorizer in the event that the user does
    # not specify one.
    if method in ("sbert", "sentence-transformers"):
        return SBERTTextVectorizer(**kwargs)
    elif method in ("imagebind",):
        return ImageBindTextVectorizer(**kwargs)
    elif method in ("sklearn", "scikit-learn"):
        return SklearnTextVectorizer(**kwargs)
    elif method in ("cohere",):
        return CohereTextVectorizer(**kwargs)
    elif method in ("voyage", "voyageai"):
        return VoyageTextVectorizer(**kwargs)
    else:
        raise NotImplementedError
