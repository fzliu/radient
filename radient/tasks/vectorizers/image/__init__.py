__all__ = [
    "TimmImageVectorizer",
    "image_vectorizer"
]

from typing import Optional

from radient.tasks.vectorizers.image._base import ImageVectorizer
from radient.tasks.vectorizers.image.timm import TimmImageVectorizer
from radient.tasks.vectorizers.image.imagebind import ImageBindImageVectorizer


def image_vectorizer(method: str = "timm", **kwargs) -> ImageVectorizer:
    """Creates an image vectorizer specified by `method`.
    """

    # Return a reasonable default vectorizer in the event that the user does
    # not specify one.
    if method in ("timm", "pytorch-image-models"):
        return TimmImageVectorizer(**kwargs)
    elif method in ("imagebind",):
        return ImageBindImageVectorizer(**kwargs)
    else:
        raise NotImplementedError
