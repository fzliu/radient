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
        self._model_name = None
        self._model = None

    def __call__(self, *args, **kwargs):
        return self.vectorize(*args, **kwargs)

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

    def _postprocess(self, vector: Vector, normalize: bool = True) -> Vector:
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
        data: Union[Any, List[Any], Dict[str, Union[Any, List[Any]]]],
        normalize: bool = True
    ) -> Union[Vector, List[Vector], Dict[str, Union[List[Vector], Vector]]]:
        """Vectorizers accept three types of inputs:

        1) One instance of the object/data,
        2) A list of data to be vectorized,
        3) A dict of {modality: data} pairs.

        This function handles all three of these cases automatically. For
        dictionary inputs, if the input modality is incapable of being handled
        by this vectorizer, the output will not include that modality.

        In a future version, the third input type (dict of modality/data pairs)
        will be replaced by a pandas dataframe to ensure better compatiblity
        with list-of-dict inputs.
        """

        def _helper(
            data_: Union[Any, List[Any]],
            modality: str = self.vtype
        ):
            single_input = False
            if not isinstance(data_, list):
                single_input = True
                data_ = [data_]

            vectors = []
            for item in data_:
                item = self._preprocess(item)
                vector = self._vectorize(item, modality=modality)
                vector = self._postprocess(vector, normalize=normalize)
                vectors.append(vector)
                vector.putmeta("data", str(item))
                vector.putmeta("modality", modality)
            if single_input:
                return vectors[0]

            return vectors

        if isinstance(data, dict):
            vectors = {}
            for modality, data_ in data.items():
                if modality in self.modalities():
                    vectors[modality] = _helper(data_, modality=modality)
            return vectors
        else:
            return _helper(data)

    def accelerate(self):
        warnings.warn("this vectorizer does not support acceleration")
