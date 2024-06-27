from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np

from radient.tasks import Task
from radient.utils import fully_qualified_name
from radient.vector import Vector


def normalize_vector(vector: Vector, inplace: bool = True):
    if not np.issubdtype(vector.dtype, np.floating):
        warnings.warn("non-float vectors are not normalized")
    else:
        if inplace:
            vector /= np.linalg.norm(vector)
        else:
            vector = vector / np.linalg.norm(vector)
    return vector


class Vectorizer(Task):
    """Base class for all vectorizers. Custom vectorizers shouldn't directly
    inherit this class, but should inherit the appropriate subclass e.g.
    `ImageVectorizer` or `AudioVectorizer`.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()
        self._model_name = None
        self._model = None

    def __call__(self, *args, **kwargs):
        vectors = self.vectorize(*args, **kwargs)
        return {"vectors": vectors}

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def model(self) -> Optional[Any]:
        return self._model

    @property
    def vtype(self) -> str:
        return fully_qualified_name(self).split(".")[3]

    def _preprocess(self, item: Any, **kwargs) -> Any:
        return item

    @abstractmethod
    def _vectorize(self, data: Any, **kwargs) -> Vector:
        pass

    def _postprocess(self, vector: Vector, normalize: bool = True, **kwargs) -> Vector:
        if normalize:
            # Some vectorizers return a _sequence_ of vectors for a single
            # piece of data (most commonly data that varies with time such as
            # videos and audio). Automatically check for these here and
            # normalize them if this is the case.
            if isinstance(vector, Sequence):
                for v in vector:
                    normalize_vector(v)
            else:
                normalize_vector(vector)
        return vector

    def modalities(self) -> List[str]:
        return [self.vtype]

    def vectorize(
        self,
        data: Union[Any, List[Any]],
        modality: Optional[str] = None,
        normalize: bool = True,
        **kwargs
    ) -> Union[Vector, List[Vector], Dict[str, Union[List[Vector], Vector]]]:
        """Vectorizers accept two types of inputs:

        1) One instance of the object/data,
        2) A list of data to be vectorized.

        This function handles both of these cases automatically.
        """
        modality = modality or self.vtype
        if modality in self.modalities():
            data_ = data if isinstance(data, list) else [data]
            vectors = []
            for d in data_:
                v = self._preprocess(d, modality=modality)
                v = self._vectorize(v, modality=modality)
                v = self._postprocess(v, modality=modality, normalize=normalize)
                # TODO(fzliu): only store the original paths, e.g. no base64
                # encodings or long-form text stored as metadata
                v.putmeta("data", str(d)).putmeta("modality", modality)
                vectors.append(v)
            return vectors[0] if not isinstance(data, list) else vectors
        else:
            warnings.warn(f"vectorizer does not support modality: {modality}")

    def accelerate(self):
        warnings.warn("vectorizer does not support acceleration")
