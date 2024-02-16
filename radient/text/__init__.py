__all__ = [
    "TextVectorizer",
    "SBERTTextVectorizer",
    "SklearnTextVectorizer",
    "text_vectorizer"
]

from typing import Optional

from radient.text.base import TextVectorizer
from radient.text.sbert import SBERTTextVectorizer
from radient.text.sklearn import SklearnTextVectorizer


def text_vectorizer(method: Optional[str] = None, **kwargs) -> TextVectorizer:
    """Creates a text vectorizer specified by `method`.
    """

    if method in (None, "sbert", "sentence-transformers"):
        if not kwargs:
            # Return a reasonable default vectorizer in the event that the user does not
            # specify one.
            return SBERTTextVectorizer("BAAI/bge-small-en-v1.5")
        else:
            return SBERTTextVectorizer(**kwargs)
    elif method in ("sklearn", "scikit-learn"):
        return SklearnTextVectorizer(**kwargs)
    else:
        raise NotImplementedError
