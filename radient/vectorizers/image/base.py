__all__ = [
    "ImageVectorizer"
]

from abc import abstractmethod
import base64
import io
from pathlib import Path
from typing import Any, List
import urllib.request

import numpy as np

from radient.util.lazy_import import fully_qualified_name, LazyImport
from radient.vectorizers.base import Vector
from radient.vectorizers.base import Vectorizer

Image = LazyImport("PIL.Image", package_name="pillow")
validators = LazyImport("validators")


class ImageVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    def _preprocess(cls, image: Any, mode: str = "RGB") -> "Image.Image":
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
        elif full_name == "builtins.str":
            # For string inputs, we support three options - a base64 encoded
            # string containing the image data, a path to a filename which is
            # a valid image type, or a URL that contains the image.
            imgpath = Path(image)
            if imgpath.suffix in Image.registered_extensions().keys():
                if imgpath.exists():
                    return Image.open(image)
                elif validators.url(image):
                    with urllib.request.urlopen(image) as resp:
                        imgbytes = io.BytesIO(resp.read())
                    return Image.open(imgbytes)
            else:
                try:
                    imgbytes = io.BytesIO(base64.b64decode(image))
                    return Image.open(imgbytes)
                except:
                    raise TypeError
        else:
            raise TypeError



