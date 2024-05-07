__all__ = [
    "TimmImageVectorizer",
    "image_vectorizer"
]

from typing import Optional

from radient.vectorizers.image.base import ImageVectorizer
from radient.vectorizers.image.timm import TimmImageVectorizer


def image_vectorizer(method: Optional[str] = None, **kwargs) -> ImageVectorizer:
    """Creates an image vectorizer specified by `method`.
    """

    # Return a reasonable default vectorizer in the event that the user does
    # not specify one.
    if method in (None, "timm", "pytorch-image-models"):
        return TimmImageVectorizer(**kwargs)
    else:
        raise NotImplementedError
