__all__ = [
    "ImageVectorizer"
]

from abc import abstractmethod
from typing import Any, List

import numpy as np

from radient.base import Vector
from radient.base import Vectorizer
from radient.util import fully_qualified_name, LazyImport

Image = LazyImport("PIL.Image", package="pillow")


class ImageVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def standardize_input(cls, image: Any, mode: str = "RGB") -> Image.Image:
        """Converts the input images into a common format, i.e. a PIL Image.
        """
        # Acquire the full class path, i.e. qualified name plus module name.
        # There might be a better way to do this that takes module rebinding
        # into consideration, but this will do for now.
        full_name = fully_qualified_name(image)
        if full_name == "PIL.Image.Image":
            return image
        elif full_name == "numpy.ndarray":
            return Image.toarray(image, mode=mode)
        else:
            raise TypeError
