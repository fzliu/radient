__all__ = [
    "TimmImageVectorizer",
    "image_vectorizer"
]

from typing import Optional

from radient.image.base import ImageVectorizer
from radient.image.timm import TimmImageVectorizer


def image_vectorizer(method: Optional[str] = None, **kwargs) -> ImageVectorizer:
    """Creates an image vectorizer specified by `method`.
    """

    if method == None:
        # Return a reasonable default vectorizer in the event that the user does not
        # specify one.
        return TimmImageVectorizer()
    elif method == "timm":
        return TimmImageVectorizer(**kwargs)
    else:
        raise NotImplementedError
